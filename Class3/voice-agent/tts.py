# tts.py
import tempfile
from pathlib import Path
from TTS.api import TTS

# Lightweight English model
tts_model = TTS(model_name="tts_models/en/ljspeech/vits", progress_bar=False, gpu=False)

def synthesize_speech(text: str) -> str:
    text = (text or "").strip() or "Hello, this is a test."

    wav_path = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name)
    tts_model.tts_to_file(text=text, file_path=str(wav_path))

    return str(wav_path)



