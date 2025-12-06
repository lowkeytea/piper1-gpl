"""Phonemization with espeak-ng via subprocess (no C extension needed)."""

import re
import subprocess
import unicodedata
from typing import List


class EspeakPhonemizer:
    """Phonemizer that uses espeak-ng via subprocess."""

    def __init__(self, espeak_data_dir=None) -> None:
        """Initialize phonemizer (espeak_data_dir ignored, uses system espeak)."""
        # Verify espeak-ng is available
        try:
            result = subprocess.run(
                ["espeak-ng", "--version"],
                capture_output=True,
                text=True,
                check=True
            )
        except FileNotFoundError:
            raise RuntimeError("espeak-ng not found. Install with: sudo apt install espeak-ng")

    def phonemize(self, voice: str, text: str) -> List[List[str]]:
        """Text to phonemes grouped by sentence."""
        # Use espeak-ng to get IPA phonemes
        # -q = quiet (no audio), -x = phonemes output, --ipa = IPA format
        try:
            result = subprocess.run(
                ["espeak-ng", "-v", voice, "-q", "--ipa", "-x", text],
                capture_output=True,
                text=True,
                check=True
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"espeak-ng failed: {e.stderr}")

        phonemes_output = result.stdout.strip()

        # Split by sentence-ending punctuation
        all_phonemes: List[List[str]] = []

        # espeak-ng outputs phonemes with spaces between words
        # and newlines between sentences/clauses
        for line in phonemes_output.split("\n"):
            line = line.strip()
            if not line:
                continue

            # Filter out (lang) switch markers
            line = re.sub(r"\([^)]+\)", "", line)

            # Decompose phonemes into UTF-8 codepoints
            # This separates accent characters into separate "phonemes"
            sentence_phonemes = list(unicodedata.normalize("NFD", line))

            if sentence_phonemes:
                all_phonemes.append(sentence_phonemes)

        # If no phonemes found, return empty list with at least structure
        if not all_phonemes:
            all_phonemes = [[]]

        return all_phonemes
