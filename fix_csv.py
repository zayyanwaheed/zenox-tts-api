import pandas as pd
df = pd.read_csv('D:/voice_dataset/metadata.csv', sep='|', header=None, names=['audio_file', 'text'])
df['audio_file'] = 'wavs/' + df['audio_file'] + '.wav'
df.to_csv('D:/voice_dataset/metadata_fixed.csv', sep='|', index=False)
print('Done')