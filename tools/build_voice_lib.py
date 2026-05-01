#!/usr/bin/env python3
"""
Automated voice library builder from public datasets.
Supports Common Voice, AISHELL, and custom audio directories.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

# Add fish-speech to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import whisper
import torchaudio
from loguru import logger

WHISPER_MODEL = "base"
VOICE_LIB_PATH = Path("/tmp/fish_speech_output/voice_presets.json")
OUTPUT_DIR = Path("/tmp/fish_speech_output/voice_samples")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_voice_lib():
    if VOICE_LIB_PATH.exists():
        with open(VOICE_LIB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_voice_lib(lib):
    with open(VOICE_LIB_PATH, "w", encoding="utf-8") as f:
        json.dump(lib, f, ensure_ascii=False, indent=4)


def transcribe_audio(audio_path, language="zh"):
    """Use Whisper to transcribe audio."""
    model = whisper.load_model(WHISPER_MODEL)
    result = model.transcribe(str(audio_path), language=language)
    return result["text"].strip()


def process_audio_file(audio_path, voice_name, language="zh"):
    """Process a single audio file and add to voice library."""
    lib = load_voice_lib()
    
    if voice_name in lib:
        logger.warning(f"Voice '{voice_name}' already exists, skipping")
        return False
    
    # Transcribe
    logger.info(f"Transcribing {audio_path}...")
    text = transcribe_audio(audio_path, language)
    
    if not text:
        logger.warning(f"No text transcribed for {audio_path}")
        return False
    
    # Copy audio to output dir
    output_path = OUTPUT_DIR / f"{voice_name}.wav"
    import shutil
    shutil.copy2(audio_path, output_path)
    
    # Add to library
    lib[voice_name] = {
        "audio_path": str(output_path),
        "reference_text": text,
        "source": str(audio_path),
        "language": language
    }
    save_voice_lib(lib)
    
    logger.info(f"Added voice '{voice_name}' with text: {text}")
    return True


def process_common_voice_chinese(data_dir, max_speakers=10):
    """Process Common Voice Chinese dataset."""
    import glob
    
    # Find all wav files in the dataset
    wav_files = glob.glob(os.path.join(data_dir, "**/*.wav"), recursive=True)
    
    logger.info(f"Found {len(wav_files)} audio files")
    
    lib = load_voice_lib()
    model = whisper.load_model(WHISPER_MODEL)
    
    processed = 0
    for i, wav_file in enumerate(wav_files):
        if processed >= max_speakers:
            break
            
        # Extract speaker ID from filename
        filename = Path(wav_file).stem
        voice_name = f"CV_CN_{filename[:8]}"
        
        if voice_name in lib:
            continue
        
        # Get duration
        try:
            wav, sr = torchaudio.load(wav_file)
            duration = wav.shape[1] / sr
            
            # Skip if too short or too long
            if duration < 2 or duration > 15:
                continue
            
            # Transcribe
            logger.info(f"[{i+1}/{len(wav_files)}] Processing {filename} ({duration:.1f}s)...")
            result = model.transcribe(wav_file, language="zh")
            text = result["text"].strip()
            
            if not text or len(text) < 5:
                continue
            
            # Save
            output_path = OUTPUT_DIR / f"{voice_name}.wav"
            import shutil
            shutil.copy2(wav_file, output_path)
            
            lib[voice_name] = {
                "audio_path": str(output_path),
                "reference_text": text,
                "source": wav_file,
                "language": "zh",
                "duration": round(duration, 2)
            }
            save_voice_lib(lib)
            processed += 1
            logger.info(f"Added: {voice_name} - {text[:30]}...")
            
        except Exception as e:
            logger.error(f"Error processing {wav_file}: {e}")
            continue
    
    logger.info(f"Processed {processed} new voices")


def process_audio_directory(dir_path, max_files=20):
    """Process all audio files in a directory."""
    import glob
    
    audio_extensions = ["*.wav", "*.mp3", "*.flac", "*.m4a"]
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(glob.glob(os.path.join(dir_path, "**", ext), recursive=True))
    
    logger.info(f"Found {len(audio_files)} audio files")
    
    lib = load_voice_lib()
    model = whisper.load_model(WHISPER_MODEL)
    
    processed = 0
    for i, audio_file in enumerate(audio_files):
        if processed >= max_files:
            break
        
        filename = Path(audio_file).stem
        voice_name = f"custom_{filename[:12]}"
        
        if voice_name in lib:
            continue
        
        try:
            wav, sr = torchaudio.load(audio_file)
            duration = wav.shape[1] / sr
            
            if duration < 2 or duration > 15:
                continue
            
            logger.info(f"[{i+1}/{len(audio_files)}] Processing {filename} ({duration:.1f}s)...")
            result = model.transcribe(audio_file, language="zh")
            text = result["text"].strip()
            
            if not text or len(text) < 5:
                continue
            
            output_path = OUTPUT_DIR / f"{voice_name}.wav"
            import shutil
            shutil.copy2(audio_file, output_path)
            
            lib[voice_name] = {
                "audio_path": str(output_path),
                "reference_text": text,
                "source": audio_file,
                "language": "zh",
                "duration": round(duration, 2)
            }
            save_voice_lib(lib)
            processed += 1
            logger.info(f"Added: {voice_name} - {text[:30]}...")
            
        except Exception as e:
            logger.error(f"Error processing {audio_file}: {e}")
            continue
    
    logger.info(f"Processed {processed} new voices")


def list_voices():
    """List all voices in the library."""
    lib = load_voice_lib()
    if not lib:
        print("No voices in library")
        return
    
    print(f"\nVoice Library ({len(lib)} voices):")
    print("-" * 60)
    for name, info in lib.items():
        text_preview = info.get("reference_text", "")[:40]
        duration = info.get("duration", "N/A")
        print(f"  {name}: {text_preview}... ({duration}s)")
    print("-" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build voice library from datasets")
    parser.add_argument("command", choices=["list", "process_dir", "process_cv"],
                       help="Command to run")
    parser.add_argument("--dir", help="Directory containing audio files")
    parser.add_argument("--max", type=int, default=10, help="Max voices to process")
    parser.add_argument("--lang", default="zh", help="Language code")
    
    args = parser.parse_args()
    
    if args.command == "list":
        list_voices()
    elif args.command == "process_dir":
        if not args.dir:
            print("Error: --dir required")
            sys.exit(1)
        process_audio_directory(args.dir, args.max)
    elif args.command == "process_cv":
        if not args.dir:
            print("Error: --dir required (path to Common Voice dataset)")
            sys.exit(1)
        process_common_voice_chinese(args.dir, args.max)
