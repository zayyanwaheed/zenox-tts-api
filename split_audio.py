import os
from pydub import AudioSegment

input_folder = 'D:/voice_dataset/raw'
output_folder = 'D:/voice_dataset/wavs'
clip_num = 1

for file in os.listdir(input_folder):
    if file.endswith('.wav'):
        print(f'Splitting {file}...')
        audio = AudioSegment.from_wav(os.path.join(input_folder, file))
        duration = len(audio)
        chunk_size = 15000  # 15 seconds per clip
        for start in range(0, duration, chunk_size):
            end = min(start + chunk_size, duration)
            chunk = audio[start:end]
            if len(chunk) >= 6000:
                out_path = os.path.join(output_folder, f'clip_{clip_num:04d}.wav')
                chunk.export(out_path, format='wav')
                clip_num += 1

print(f'Done! Total clips: {clip_num-1}')