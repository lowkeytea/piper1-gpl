"""
PyTorch Lightning module with Speaker Encoder for zero-shot voice cloning.

Extends VitsModel to add:
1. ECAPA-TDNN speaker encoder
2. Speaker verification loss (AAM-Softmax)
3. Support for training with reference audio embeddings
4. Multiple training phases (encoder-only, joint, fine-tune)

Training Phases:
    Phase 1: Speaker encoder pretraining (optional)
        - Train only speaker encoder with AAM-Softmax loss
        - Requires labeled speaker data
        - Can skip if using pretrained encoder

    Phase 2: Joint training
        - Train all components together
        - Speaker encoder provides conditioning
        - Combined TTS + speaker verification loss

    Phase 3: Fine-tuning (optional)
        - Freeze speaker encoder
        - Fine-tune VITS on new voices with few samples

Usage:
    # Create model for joint training
    model = VitsModelWithSpeakerEncoder(
        num_speakers=1000,  # For AAM-Softmax during training
        use_speaker_encoder=True,
        freeze_speaker_encoder=False,
    )

    # Create model for inference only (frozen encoder)
    model = VitsModelWithSpeakerEncoder(
        use_speaker_encoder=True,
        freeze_speaker_encoder=True,
    )

    # Inference with reference audio
    ref_mel = ...  # [1, 80, time] reference mel spectrogram
    text = ...  # [1, text_len] phoneme IDs
    audio = model.infer_with_reference(text, text_lengths, ref_mel, scales)
"""

import ast
import logging
import operator
from functools import reduce
from typing import Optional, Tuple

import lightning as L
import torch
from torch import autocast
from torch.nn import functional as F

from .commons import slice_segments
from .dataset import Batch
from .losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from .mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from .models import MultiPeriodDiscriminator, SynthesizerTrn
from .speaker_encoder import SpeakerEncoder

_LOGGER = logging.getLogger(__name__)


class VitsModelWithSpeakerEncoder(L.LightningModule):
    """
    VITS model with integrated speaker encoder for zero-shot voice cloning.

    This extends the base VitsModel to support:
    1. External speaker embeddings from reference audio
    2. Joint training with speaker verification loss
    3. Multiple training phases for different use cases

    Args:
        # Speaker encoder parameters
        use_speaker_encoder: Whether to use external speaker encoder
        freeze_speaker_encoder: Whether to freeze speaker encoder weights
        speaker_encoder_loss_weight: Weight for AAM-Softmax loss during training
        speaker_encoder_pretrained: Path to pretrained speaker encoder weights

        # All other parameters from VitsModel...
    """

    def __init__(
        self,
        batch_size: int = 32,
        sample_rate: int = 22050,
        num_symbols: int = 256,
        num_speakers: int = 1,
        num_emotions: int = 0,
        # audio
        resblock="2",
        resblock_kernel_sizes=(3, 5, 7),
        resblock_dilation_sizes=(
            (1, 2),
            (2, 6),
            (3, 12),
        ),
        upsample_rates=(8, 8, 4),
        upsample_initial_channel=256,
        upsample_kernel_sizes=(16, 16, 8),
        # mel
        filter_length: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        mel_channels: int = 80,
        mel_fmin: float = 0.0,
        mel_fmax: Optional[float] = None,
        # model
        inter_channels: int = 192,
        hidden_channels: int = 192,
        filter_channels: int = 768,
        n_heads: int = 2,
        n_layers: int = 6,
        kernel_size: int = 3,
        p_dropout: float = 0.1,
        n_layers_q: int = 3,
        use_spectral_norm: bool = False,
        gin_channels: int = 0,
        emo_channels: int = 0,
        use_sdp: bool = True,
        segment_size: int = 8192,
        # training
        learning_rate: float = 2e-4,
        learning_rate_d: float = 1e-4,
        learning_rate_spk: float = 1e-3,  # Speaker encoder learning rate
        betas: tuple[float, float] = (0.8, 0.99),
        betas_d: tuple[float, float] = (0.5, 0.9),
        eps: float = 1e-9,
        lr_decay: float = 0.999875,
        lr_decay_d: float = 0.9999,
        lr_decay_spk: float = 0.99,
        init_lr_ratio: float = 1.0,
        warmup_epochs: int = 0,
        c_mel: int = 45,
        c_kl: float = 1.0,
        grad_clip: Optional[float] = None,
        # Speaker encoder parameters
        use_speaker_encoder: bool = True,
        freeze_speaker_encoder: bool = False,
        speaker_encoder_loss_weight: float = 0.1,
        speaker_consistency_loss_weight: float = 0.0,
        speaker_encoder_pretrained: Optional[str] = None,
        speaker_encoder_aam_margin: float = 0.2,
        speaker_encoder_aam_scale: float = 30.0,
        # unused
        dataset: object = None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Parse string tuples if needed
        if isinstance(self.hparams.resblock_kernel_sizes, str):
            self.hparams.resblock_kernel_sizes = ast.literal_eval(
                self.hparams.resblock_kernel_sizes
            )

        if isinstance(self.hparams.resblock_dilation_sizes, str):
            self.hparams.resblock_dilation_sizes = ast.literal_eval(
                self.hparams.resblock_dilation_sizes
            )

        if isinstance(self.hparams.upsample_rates, str):
            self.hparams.upsample_rates = ast.literal_eval(self.hparams.upsample_rates)

        if isinstance(self.hparams.upsample_kernel_sizes, str):
            self.hparams.upsample_kernel_sizes = ast.literal_eval(
                self.hparams.upsample_kernel_sizes
            )

        if isinstance(self.hparams.betas, str):
            self.hparams.betas = ast.literal_eval(self.hparams.betas)

        expected_hop_length = reduce(operator.mul, self.hparams.upsample_rates, 1)
        if expected_hop_length != hop_length:
            raise ValueError("Upsample rates do not match hop length")

        # Need to use manual optimization because we have multiple optimizers
        self.automatic_optimization = False

        self.batch_size = batch_size

        # Set gin_channels for speaker encoder output
        if use_speaker_encoder and self.hparams.gin_channels <= 0:
            self.hparams.gin_channels = 512

        # Auto-set emotion channels if emotions are enabled
        if (self.hparams.num_emotions > 1) and (self.hparams.emo_channels <= 0):
            self.hparams.emo_channels = 256

        # Create speaker encoder
        self.speaker_encoder: Optional[SpeakerEncoder] = None
        if use_speaker_encoder:
            self.speaker_encoder = SpeakerEncoder(
                gin_channels=self.hparams.gin_channels,
                n_mels=mel_channels,
                freeze_ecapa=freeze_speaker_encoder,
                n_speakers=num_speakers if num_speakers > 1 else 0,
                aam_margin=speaker_encoder_aam_margin,
                aam_scale=speaker_encoder_aam_scale,
            )

            # Load pretrained weights if provided
            if speaker_encoder_pretrained:
                self.speaker_encoder.load_pretrained_ecapa(speaker_encoder_pretrained)

        # Set up VITS models
        self.model_g = SynthesizerTrn(
            n_vocab=num_symbols,
            spec_channels=self.hparams.filter_length // 2 + 1,
            segment_size=self.hparams.segment_size // self.hparams.hop_length,
            inter_channels=self.hparams.inter_channels,
            hidden_channels=self.hparams.hidden_channels,
            filter_channels=self.hparams.filter_channels,
            n_heads=self.hparams.n_heads,
            n_layers=self.hparams.n_layers,
            kernel_size=self.hparams.kernel_size,
            p_dropout=self.hparams.p_dropout,
            resblock=self.hparams.resblock,
            resblock_kernel_sizes=self.hparams.resblock_kernel_sizes,
            resblock_dilation_sizes=self.hparams.resblock_dilation_sizes,
            upsample_rates=self.hparams.upsample_rates,
            upsample_initial_channel=self.hparams.upsample_initial_channel,
            upsample_kernel_sizes=self.hparams.upsample_kernel_sizes,
            n_speakers=1 if use_speaker_encoder else num_speakers,  # No embedding table needed
            n_emotions=self.hparams.num_emotions,
            gin_channels=self.hparams.gin_channels,
            emo_channels=self.hparams.emo_channels,
            use_sdp=self.hparams.use_sdp,
            use_external_speaker_emb=use_speaker_encoder,
        )

        self.model_d = MultiPeriodDiscriminator(
            use_spectral_norm=self.hparams.use_spectral_norm
        )

    def _extract_reference_mel(self, batch: Batch) -> torch.Tensor:
        """
        Extract reference mel spectrogram from batch audio.

        For training, we use the target audio as the reference.
        In practice, you'd want a separate reference audio.

        Args:
            batch: Training batch

        Returns:
            Reference mel spectrogram [batch, n_mels, time]
        """
        # Use the spectrogram from batch directly
        spec = batch.spectrograms
        mel = spec_to_mel_torch(
            spec,
            self.hparams.filter_length,
            self.hparams.mel_channels,
            self.hparams.sample_rate,
            self.hparams.mel_fmin,
            self.hparams.mel_fmax,
        )
        return mel

    def forward(
        self,
        text,
        text_lengths,
        scales,
        sid=None,
        eid=None,
        ref_mel=None,
    ):
        """
        Forward pass for inference.

        Args:
            text: Phoneme IDs [batch, text_len]
            text_lengths: Text lengths [batch]
            scales: [noise_scale, length_scale, noise_scale_w]
            sid: Speaker ID (only used if speaker_encoder disabled)
            eid: Emotion ID
            ref_mel: Reference mel spectrogram [batch, n_mels, time]

        Returns:
            Generated audio [batch, 1, time]
        """
        noise_scale = scales[0]
        length_scale = scales[1]
        noise_scale_w = scales[2]

        # Get speaker conditioning
        g_external = None
        if self.speaker_encoder is not None and ref_mel is not None:
            g_external, _ = self.speaker_encoder(ref_mel)

        audio, *_ = self.model_g.infer(
            text,
            text_lengths,
            noise_scale=noise_scale,
            length_scale=length_scale,
            noise_scale_w=noise_scale_w,
            sid=sid,
            eid=eid,
            g_external=g_external,
        )

        return audio

    def infer_with_reference(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        ref_mel: torch.Tensor,
        scales: Tuple[float, float, float] = (0.667, 1.0, 0.8),
        eid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Synthesize audio using reference audio for speaker embedding.

        Args:
            text: Phoneme IDs [batch, text_len]
            text_lengths: Text lengths [batch]
            ref_mel: Reference mel spectrogram [batch, n_mels, time]
            scales: (noise_scale, length_scale, noise_scale_w)
            eid: Optional emotion ID [batch]

        Returns:
            Generated audio [batch, 1, time]
        """
        assert self.speaker_encoder is not None, "Speaker encoder not initialized"

        # Extract speaker embedding from reference
        g_external, _ = self.speaker_encoder(ref_mel)

        # Generate audio
        audio, *_ = self.model_g.infer(
            text,
            text_lengths,
            noise_scale=scales[0],
            length_scale=scales[1],
            noise_scale_w=scales[2],
            eid=eid,
            g_external=g_external,
        )

        return audio

    def _compute_loss(self, batch: Batch):
        """Compute generator and discriminator losses."""
        x, x_lengths, y, _, spec, spec_lengths, speaker_ids, emotion_ids = (
            batch.phoneme_ids,
            batch.phoneme_lengths,
            batch.audios,
            batch.audio_lengths,
            batch.spectrograms,
            batch.spectrogram_lengths,
            batch.speaker_ids if batch.speaker_ids is not None else None,
            batch.emotion_ids if batch.emotion_ids is not None else None,
        )

        # Get speaker conditioning from reference audio
        g_external = None
        spk_loss = None
        if self.speaker_encoder is not None:
            ref_mel = self._extract_reference_mel(batch)
            g_external, spk_loss = self.speaker_encoder(ref_mel, speaker_ids)

        # Forward through generator
        (
            y_hat,
            l_length,
            _attn,
            ids_slice,
            _x_mask,
            z_mask,
            (_z, z_p, m_p, logs_p, _m_q, logs_q),
        ) = self.model_g(
            x, x_lengths, spec, spec_lengths,
            sid=None if self.speaker_encoder else speaker_ids,
            eid=emotion_ids,
            g_external=g_external,
        )

        # Compute mel spectrograms
        mel = spec_to_mel_torch(
            spec,
            self.hparams.filter_length,
            self.hparams.mel_channels,
            self.hparams.sample_rate,
            self.hparams.mel_fmin,
            self.hparams.mel_fmax,
        )
        y_mel = slice_segments(
            mel,
            ids_slice,
            self.hparams.segment_size // self.hparams.hop_length,
        )
        y_hat_mel = mel_spectrogram_torch(
            y_hat.squeeze(1),
            self.hparams.filter_length,
            self.hparams.mel_channels,
            self.hparams.sample_rate,
            self.hparams.hop_length,
            self.hparams.win_length,
            self.hparams.mel_fmin,
            self.hparams.mel_fmax,
        )
        y = slice_segments(
            y,
            ids_slice * self.hparams.hop_length,
            self.hparams.segment_size,
        )

        # Trim to avoid padding issues
        y_hat = y_hat[..., : y.shape[-1]]

        _y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = self.model_d(y, y_hat)

        with autocast(self.device.type, enabled=False):
            # Generator loss
            loss_dur = torch.sum(l_length.float())
            loss_mel = F.l1_loss(y_mel, y_hat_mel) * self.hparams.c_mel
            loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * self.hparams.c_kl

            loss_fm = feature_loss(fmap_r, fmap_g)
            loss_gen, _losses_gen = generator_loss(y_d_hat_g)
            loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl

            # Add speaker encoder loss if enabled
            if spk_loss is not None:
                loss_gen_all = loss_gen_all + spk_loss * self.hparams.speaker_encoder_loss_weight

            # Speaker consistency loss: re-encode synthesized audio and match the
            # (detached) reference speaker conditioning.
            spk_consistency_loss = None
            if (
                self.speaker_encoder is not None
                and getattr(self.hparams, "speaker_consistency_loss_weight", 0.0) > 0.0
                and g_external is not None
            ):
                g_ref = g_external.detach().squeeze(-1).float()
                g_synth, _ = self.speaker_encoder(
                    y_hat_mel.float(),
                    speaker_labels=None,
                    enable_grad=True,
                )
                g_synth = g_synth.squeeze(-1).float()
                spk_consistency_loss = 1.0 - F.cosine_similarity(g_synth, g_ref, dim=1).mean()
                loss_gen_all = loss_gen_all + spk_consistency_loss * self.hparams.speaker_consistency_loss_weight

        # Discriminator step
        y_d_hat_r, y_d_hat_g, _, _ = self.model_d(y, y_hat.detach())

        with autocast(self.device.type, enabled=False):
            loss_disc, _losses_disc_r, _losses_disc_g = discriminator_loss(
                y_d_hat_r, y_d_hat_g
            )
            loss_disc_all = loss_disc

        return loss_gen_all, loss_disc_all, spk_loss, spk_consistency_loss

    def training_step(self, batch: Batch, batch_idx: int):
        opt_g, opt_d = self.optimizers()[:2]
        loss_g, loss_d, spk_loss, spk_consistency_loss = self._compute_loss(batch)

        # Log losses
        self.log("loss_g", loss_g, batch_size=self.batch_size)
        if spk_loss is not None:
            self.log("loss_spk", spk_loss, batch_size=self.batch_size)
        if spk_consistency_loss is not None:
            self.log("loss_spk_consistency", spk_consistency_loss, batch_size=self.batch_size)

        # Generator step
        opt_g.zero_grad()
        self.manual_backward(loss_g, retain_graph=True)
        opt_g.step()

        # Discriminator step
        self.log("loss_d", loss_d, batch_size=self.batch_size)
        opt_d.zero_grad()
        self.manual_backward(loss_d)
        opt_d.step()

    def validation_step(self, batch: Batch, batch_idx: int):
        loss_g, _loss_d, spk_loss, spk_consistency_loss = self._compute_loss(batch)
        val_loss = loss_g
        self.log("val_loss", val_loss, batch_size=self.batch_size)
        if spk_loss is not None:
            self.log("val_loss_spk", spk_loss, batch_size=self.batch_size)
        if spk_consistency_loss is not None:
            self.log("val_loss_spk_consistency", spk_consistency_loss, batch_size=self.batch_size)
        return val_loss

    def on_validation_end(self) -> None:
        """Generate audio examples after validation."""
        if self.trainer.sanity_checking:
            return super().on_validation_end()

        if (
            getattr(self, "logger", None)
            and hasattr(self.logger, "experiment")
            and hasattr(self.logger.experiment, "add_audio")
        ):
            for utt_idx, test_utt in enumerate(self.trainer.datamodule.test_dataset):
                text = test_utt.phoneme_ids.unsqueeze(0).to(self.device)
                text_lengths = torch.LongTensor([len(test_utt.phoneme_ids)]).to(
                    self.device
                )
                scales = [0.667, 1.0, 0.8]

                # Get reference mel from test utterance audio (or cached reference spec)
                ref_mel = None
                if self.speaker_encoder is not None:
                    if test_utt.reference_spec is not None:
                        # Cached linear reference spectrogram from manifest → mel
                        ref_spec = test_utt.reference_spec.unsqueeze(0).to(self.device)
                        ref_mel = spec_to_mel_torch(
                            ref_spec,
                            self.hparams.filter_length,
                            self.hparams.mel_channels,
                            self.hparams.sample_rate,
                            self.hparams.mel_fmin,
                            self.hparams.mel_fmax,
                        )
                    else:
                        # Compute mel from normalized audio
                        ref_audio = test_utt.audio_norm.unsqueeze(0).to(self.device)
                        ref_mel = mel_spectrogram_torch(
                            ref_audio,
                            self.hparams.filter_length,
                            self.hparams.mel_channels,
                            self.hparams.sample_rate,
                            self.hparams.hop_length,
                            self.hparams.win_length,
                            self.hparams.mel_fmin,
                            self.hparams.mel_fmax,
                        )

                sid = (
                    test_utt.speaker_id.to(self.device)
                    if test_utt.speaker_id is not None
                    else None
                )
                eid = (
                    test_utt.emotion_id.to(self.device)
                    if test_utt.emotion_id is not None
                    else None
                )

                test_audio = self(
                    text, text_lengths, scales, sid=sid, eid=eid, ref_mel=ref_mel
                ).detach()

                # Scale to make louder in [-1, 1]
                test_audio = test_audio * (1.0 / max(0.01, abs(test_audio).max()))

                tag = test_utt.text or str(utt_idx)
                self.logger.experiment.add_audio(
                    tag, test_audio, sample_rate=self.hparams.sample_rate
                )

        return super().on_validation_end()

    def configure_optimizers(self):
        # Collect generator parameters (VITS + speaker encoder projection)
        gen_params = list(self.model_g.parameters())
        if self.speaker_encoder is not None:
            # Add projection layer parameters
            gen_params.extend(self.speaker_encoder.projection.parameters())
            # Add ECAPA parameters if not frozen
            if not self.hparams.freeze_speaker_encoder:
                gen_params.extend(self.speaker_encoder.ecapa.parameters())
            # Add AAM-Softmax if present
            if self.speaker_encoder.aam_softmax is not None:
                gen_params.extend(self.speaker_encoder.aam_softmax.parameters())

        optimizers = [
            torch.optim.AdamW(
                gen_params,
                lr=self.hparams.learning_rate,
                betas=self.hparams.betas,
                eps=self.hparams.eps,
            ),
            torch.optim.AdamW(
                self.model_d.parameters(),
                lr=self.hparams.learning_rate_d,
                betas=self.hparams.betas_d,
                eps=self.hparams.eps,
            ),
        ]

        schedulers = [
            torch.optim.lr_scheduler.ExponentialLR(
                optimizers[0], gamma=self.hparams.lr_decay
            ),
            torch.optim.lr_scheduler.ExponentialLR(
                optimizers[1], gamma=self.hparams.lr_decay_d
            ),
        ]

        return optimizers, schedulers

    def voice_conversion(
        self,
        source_audio: torch.Tensor,
        source_ref_mel: torch.Tensor,
        target_ref_mel: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert voice from source speaker to target speaker.

        Args:
            source_audio: Source audio [batch, time]
            source_ref_mel: Source speaker reference mel [batch, n_mels, time]
            target_ref_mel: Target speaker reference mel [batch, n_mels, time]

        Returns:
            Converted audio [batch, 1, time]
        """
        assert self.speaker_encoder is not None, "Speaker encoder not initialized"

        # Extract speaker embeddings
        g_src, _ = self.speaker_encoder(source_ref_mel)
        g_tgt, _ = self.speaker_encoder(target_ref_mel)

        # Compute spectrogram from source audio
        source_spec = mel_spectrogram_torch(
            source_audio,
            self.hparams.filter_length,
            self.hparams.mel_channels,
            self.hparams.sample_rate,
            self.hparams.hop_length,
            self.hparams.win_length,
            self.hparams.mel_fmin,
            self.hparams.mel_fmax,
        )

        # Convert
        y_lengths = torch.LongTensor([source_spec.size(-1)]).to(source_spec.device)
        o_hat, *_ = self.model_g.voice_conversion(
            source_spec.unsqueeze(1),
            y_lengths,
            g_src=g_src,
            g_tgt=g_tgt,
        )

        return o_hat
