import whisper
import os

model = whisper.load_model("base")
clips_folder = "D:/voice_dataset/wavs"
output_file = "D:/voice_dataset/metadata.csv"

with open(output_file, "w", encoding="utf-8") as f:
    for file in sorted(os.listdir(clips_folder)):
        if file.endswith(".wav"):
            path = os.path.join(clips_folder, file)
            print(f"Transcribing {file}...")
            result = model.transcribe(path, language="zh")
            text = result["text"].strip()
            clip_name = file.replace(".wav", "")
            f.write(f"{clip_name}|{text}\n")
            print(f"  → {text}")

print("Done! Saved to metadata.csv")