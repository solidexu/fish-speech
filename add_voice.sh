#!/bin/bash
# Add a new voice to Fish Speech reference library
# Usage: ./add_voice.sh <audio_path> <voice_name>

set -e

AUDIO_PATH="$1"
VOICE_NAME="$2"

if [ -z "$AUDIO_PATH" ] || [ -z "$VOICE_NAME" ]; then
    echo "Usage: $0 <audio_path> <voice_name>"
    echo "Example: $0 /path/to/voice.wav young_female_1"
    exit 1
fi

if [ ! -f "$AUDIO_PATH" ]; then
    echo "Error: Audio file not found: $AUDIO_PATH"
    exit 1
fi

REF_DIR="/disk0/repo/manju/fish-speech/references"
WHISHER_VENV="/disk0/repo/manju/fish-speech/.venv/bin/python3"

# Create reference directory
VOICE_DIR="$REF_DIR/$VOICE_NAME"
if [ -d "$VOICE_DIR" ]; then
    echo "Error: Voice already exists: $VOICE_NAME"
    exit 1
fi

mkdir -p "$VOICE_DIR"

# Copy audio as sample.wav
cp "$AUDIO_PATH" "$VOICE_DIR/sample.wav"
echo "Copied audio to $VOICE_DIR/sample.wav"

# Transcribe with Whisper
echo "Transcribing audio..."
TRANSCRIPT=$($WHISHER_VENV -c "
import whisper
model = whisper.load_model('base')
result = model.transcribe('$AUDIO_PATH', language='zh')
print(result['text'].strip())
")

echo "Transcript: $TRANSCRIPT"

# Create .lab file
echo "$TRANSCRIPT" > "$VOICE_DIR/sample.lab"
echo "Created $VOICE_DIR/sample.lab"

echo "Voice '$VOICE_NAME' added successfully!"
