import os
import pandas as pd
import librosa

folder_path = "C:\\Study\\Sem 3\\DS203\\Project\\Bhav Geet\\Marathi"  # Change path
output_path = "C:\\Study\\Sem 3\\DS203\\Project\\Bhav Geet Data"  # Change output path
# 0-National Anthem, 1-Bhav Geet, 2-Lavani songs,
# 3-Asha Bhosale, 4-Kishor Kumar, 5-Michael Jackson
label = 1  # Change label accordingly

song_num = 1
data = pd.DataFrame(columns=("song_id", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19))

def create_MFCC_coefficients(file_name, label, song_num):
    sr_value = 44100
    n_mfcc_count = 20

    try:
        y, sr = librosa.load(file_name, sr=sr_value)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc_count)
        coeff_df = (pd.DataFrame(mfccs)).transpose()
        coeff_df.insert(0, 'song_id', str(song_num))
        coeff_df.insert(1, 'label', str(label))
        return coeff_df
    except Exception as e:
        print(f"Error creating MFCC coefficients: {file_name}:{str(e)}")

for song in os.listdir(folder_path):
    song_path = (folder_path + "\\" + song)
    coeff_df = create_MFCC_coefficients(song_path, label, song_num)
    song_num += 1
    data = pd.concat([data, coeff_df], axis=0, ignore_index=True)
    print(song, "done.")

print(data)
data.to_csv(f"{output_path}\\data{label}.csv")