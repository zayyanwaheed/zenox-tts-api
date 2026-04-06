import soundfile as sf
import torch, torchaudio
def _load(path, *a, **kw):
    d, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return torch.from_numpy(d.T), sr
torchaudio.load = _load

from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException
from fastapi.responses import FileResponse
import uvicorn
import tempfile
import numpy as np
import scipy.io.wavfile as wav

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

API_KEY = "znx-dccdcb1c7c1b05efbef3e2b9ffd4f9b1"

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

@app.post("/synthesize")
async def synthesize(
    text: str = Form(...),
    language: str = Form(default="zh-cn"),
    speaker_wav: UploadFile = File(default=None),
    x_api_key: str = Header(...)
):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if speaker_wav:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(await speaker_wav.read())
            speaker_path = f.name
    else:
        speaker_path = "D:/voice_dataset/wavs/clip_0001.wav"

    outputs = model.synthesize(
        text=text,
        config=config,
        speaker_wav=speaker_path,
        language=language,
    )

    out_path = tempfile.mktemp(suffix=".wav")
    wav.write(out_path, 24000, np.array(outputs["wav"]))
    return FileResponse(out_path, media_type="audio/wav", filename="output.wav")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)