#!/usr/bin/env python3

import argparse
import logging
from pathlib import Path
from typing import Optional

import torch

from .vits.lightning import VitsModel

_LOGGER = logging.getLogger(__name__)
OPSET_VERSION = 15


def main() -> None:
    """Main entry point"""
    torch.manual_seed(1234)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", required=True, help="Path to model checkpoint (.ckpt)"
    )
    parser.add_argument(
        "--output-file", required=True, help="Path to output file (.onnx)"
    )

    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to the console"
    )
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    _LOGGER.debug(args)

    # -------------------------------------------------------------------------

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(args.checkpoint)

    # pylint: disable=no-value-for-parameter
    model = VitsModel.load_from_checkpoint(checkpoint_path, map_location="cpu")
    model_g = model.model_g

    # Inference only
    model_g.eval()

    with torch.no_grad():
        model_g.dec.remove_weight_norm()

    num_symbols = model_g.n_vocab
    num_speakers = model_g.n_speakers
    num_emotions = model_g.n_emotions

    has_speakers = num_speakers > 1
    has_emotions = num_emotions > 1

    # Build forward wrapper with correct signature based on model config
    # ONNX tracing requires positional args, so we can't use **kwargs
    if has_speakers and has_emotions:
        def infer_forward(text, text_lengths, scales, sid, eid):
            audio = model_g.infer(
                text, text_lengths,
                noise_scale=scales[0], length_scale=scales[1], noise_scale_w=scales[2],
                sid=sid, eid=eid,
            )[0].unsqueeze(1)
            return audio
    elif has_speakers:
        def infer_forward(text, text_lengths, scales, sid):
            audio = model_g.infer(
                text, text_lengths,
                noise_scale=scales[0], length_scale=scales[1], noise_scale_w=scales[2],
                sid=sid,
            )[0].unsqueeze(1)
            return audio
    elif has_emotions:
        def infer_forward(text, text_lengths, scales, eid):
            audio = model_g.infer(
                text, text_lengths,
                noise_scale=scales[0], length_scale=scales[1], noise_scale_w=scales[2],
                eid=eid,
            )[0].unsqueeze(1)
            return audio
    else:
        def infer_forward(text, text_lengths, scales):
            audio = model_g.infer(
                text, text_lengths,
                noise_scale=scales[0], length_scale=scales[1], noise_scale_w=scales[2],
            )[0].unsqueeze(1)
            return audio

    model_g.forward = infer_forward  # type: ignore[method-assign,assignment]

    # Build dummy inputs
    dummy_input_length = 50
    sequences = torch.randint(
        low=0, high=num_symbols, size=(1, dummy_input_length), dtype=torch.long
    )
    sequence_lengths = torch.LongTensor([sequences.size(1)])
    scales = torch.FloatTensor([0.667, 1.0, 0.8])

    # Build input list and names based on model config
    dummy_input_list = [sequences, sequence_lengths, scales]
    input_names = ["input", "input_lengths", "scales"]

    if has_speakers:
        dummy_input_list.append(torch.LongTensor([0]))
        input_names.append("sid")

    if has_emotions:
        dummy_input_list.append(torch.LongTensor([0]))
        input_names.append("eid")

    dummy_input = tuple(dummy_input_list)

    # Export
    torch.onnx.export(
        model=model_g,
        args=dummy_input,
        f=output_path,
        verbose=False,
        opset_version=OPSET_VERSION,
        input_names=input_names,  # NEW: use dynamic list
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 1: "phonemes"},
            "input_lengths": {0: "batch_size"},
            "output": {0: "batch_size", 2: "time"},
        },
    )
    _LOGGER.info("Exported model to %s", output_path)
    _LOGGER.info(
        "Model info: %d symbols, %d speakers, %d emotions",
        num_symbols, num_speakers, num_emotions
    )  # NEW: log emotion count


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
