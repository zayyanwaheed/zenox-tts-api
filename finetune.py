import os
os.environ["COQUI_TTS_HOME"] = "C:/Users/ZENOX/AppData/Local/tts"

import soundfile as sf
import torch, torchaudio
def _load(path, *a, **kw):
    d, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return torch.from_numpy(d.T), sr
torchaudio.load = _load

if __name__ == "__main__":
    from TTS.demos.xtts_ft_demo.utils.gpt_train import train_gpt
    train_gpt(
        language="zh-cn",
        train_csv="D:/voice_dataset/metadata_fixed.csv",
        eval_csv="D:/voice_dataset/metadata_fixed.csv",
        num_epochs=5,
        batch_size=2,
        grad_acumm=1,
        output_path="D:/voice_dataset/trained",
        max_audio_length=255995,
    )
    print("Training complete!")