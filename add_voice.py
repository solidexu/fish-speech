#!/usr/bin/env python3
"""Add a new voice to Fish Speech reference library.

Usage:
    python add_voice.py <audio_path> <voice_name>

Example:
    python add_voice.py /path/to/voice.wav young_female_1
"""

import os
import shutil
import sys
from pathlib import Path

# Add fish-speech to path
sys.path.insert(0, str(Path(__file__).parent))

import whisper

REF_DIR = Path(__file__).parent / "references"
REF_DIR.mkdir(exist_ok=True)


def add_voice(audio_path: str, voice_name: str) -> None:
    """Add a new voice to the reference library."""
    audio_path = Path(audio_path)
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        sys.exit(1)

    voice_dir = REF_DIR / voice_name
    if voice_dir.exists():
        print(f"Error: Voice already exists: {voice_name}")
        sys.exit(1)

    # Create reference directory
    voice_dir.mkdir(parents=True, exist_ok=True)

    # Copy audio as sample.wav
    shutil.copy2(audio_path, voice_dir / "sample.wav")
    print(f"Copied audio to {voice_dir / 'sample.wav'}")

    # Transcribe with Whisper
    print("Transcribing audio...")
    model = whisper.load_model("base")
    result = model.transcribe(str(audio_path), language="zh")
    transcript = result["text"].strip()
    print(f"Transcript: {transcript}")

    # Create .lab file
    (voice_dir / "sample.lab").write_text(transcript, encoding="utf-8")
    print(f"Created {voice_dir / 'sample.lab'}")

    print(f"\nVoice '{voice_name}' added successfully!")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python add_voice.py <audio_path> <voice_name>")
        sys.exit(1)

    add_voice(sys.argv[1], sys.argv[2])
