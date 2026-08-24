import whisper
import soundfile as sf

model = whisper.load_model(
    "large-v3",
    download_root="models/speech_encoder/"
)


