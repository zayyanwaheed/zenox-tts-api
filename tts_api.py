import soundfile as sf
import torch, torchaudio
def _load(path, *a, **kw):
    d, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return torch.from_numpy(d.T), sr
torchaudio.load = _load

from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tempfile
import numpy as np
import scipy.io.wavfile as wav
import os
from pydub import AudioSegment
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from datetime import datetime, timedelta
from collections import defaultdict
import logging

# Logging setup
logging.basicConfig(
    filename="D:/tts_requests.log",
    level=logging.INFO,
    format="%(asctime)s | %(message)s"
)

API_KEY = "znx-dccdcb1c7c1b05efbef3e2b9ffd4f9b1"

# Rate limiting: max 10 requests per minute per IP
RATE_LIMIT = 10
rate_tracker = defaultdict(list)

def is_rate_limited(ip):
    now = datetime.now()
    window = now - timedelta(minutes=1)
    rate_tracker[ip] = [t for t in rate_tracker[ip] if t > window]
    if len(rate_tracker[ip]) >= RATE_LIMIT:
        return True
    rate_tracker[ip].append(now)
    return False

# Emotion presets
EMOTION_PRESETS = {
    "neutral":   {"temperature": 0.65, "speed": 1.0,  "repetition_penalty": 2.0},
    "happy":     {"temperature": 0.85, "speed": 1.15, "repetition_penalty": 2.5},
    "sad":       {"temperature": 0.55, "speed": 0.85, "repetition_penalty": 1.8},
    "angry":     {"temperature": 0.90, "speed": 1.2,  "repetition_penalty": 3.0},
    "calm":      {"temperature": 0.50, "speed": 0.90, "repetition_penalty": 1.5},
    "excited":   {"temperature": 0.95, "speed": 1.3,  "repetition_penalty": 2.8},
}

config = XttsConfig()
config.load_json("D:/voice_dataset/trained/run/training/GPT_XTTS_FT-April-05-2026_10+18PM-0000000/config.json")
model = Xtts.init_from_config(config)
model.load_checkpoint(
    config,
    checkpoint_dir="D:/voice_dataset/trained/run/training/GPT_XTTS_FT-April-05-2026_10+18PM-0000000",
    vocab_path="D:/voice_dataset/trained/run/training/XTTS_v2.0_original_model_files/vocab.json",
    eval=True
)
model.cpu()

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def convert_to_wav(input_path):
    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".mp3":
        wav_path = input_path.replace(".mp3", ".wav")
        AudioSegment.from_mp3(input_path).export(wav_path, format="wav")
        return wav_path
    return input_path

@app.get("/emotions")
def get_emotions():
    return {"emotions": list(EMOTION_PRESETS.keys())}

@app.post("/synthesize")
async def synthesize(
    request: Request,
    text: str = Form(...),
    language: str = Form(default="zh-cn"),
    speaker_wav: UploadFile = File(default=None),
    output_format: str = Form(default="wav"),
    emotion: str = Form(default="neutral"),
    x_api_key: str = Header(...)
):
    ip = request.client.host

    if x_api_key != API_KEY:
        logging.warning(f"INVALID KEY | IP: {ip} | key: {x_api_key}")
        raise HTTPException(status_code=401, detail="Invalid API key")

    if is_rate_limited(ip):
        logging.warning(f"RATE LIMITED | IP: {ip}")
        raise HTTPException(status_code=429, detail="Too many requests. Max 10 per minute.")

    # Get emotion settings
    preset = EMOTION_PRESETS.get(emotion, EMOTION_PRESETS["neutral"])
    logging.info(f"REQUEST | IP: {ip} | lang: {language} | format: {output_format} | emotion: {emotion} | text: {text[:50]}")

    if speaker_wav:
        ext = os.path.splitext(speaker_wav.filename)[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as f:
            f.write(await speaker_wav.read())
            speaker_path = f.name
        speaker_path = convert_to_wav(speaker_path)
    else:
        speaker_path = "D:/voice_dataset/wavs/clip_0001.wav"

    outputs = model.synthesize(
        text=text,
        config=config,
        speaker_wav=speaker_path,
        language=language,
        temperature=preset["temperature"],
        speed=preset["speed"],
        repetition_penalty=preset["repetition_penalty"],
    )

    out_wav = tempfile.mktemp(suffix=".wav")
    wav.write(out_wav, 24000, np.array(outputs["wav"]))

    logging.info(f"SUCCESS | IP: {ip} | lang: {language} | emotion: {emotion}")

    if output_format == "mp3":
        out_mp3 = out_wav.replace(".wav", ".mp3")
        AudioSegment.from_wav(out_wav).export(out_mp3, format="mp3")
        return FileResponse(out_mp3, media_type="audio/mpeg", filename="output.mp3")

    return FileResponse(out_wav, media_type="audio/wav", filename="output.wav")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)