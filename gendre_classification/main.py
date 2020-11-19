import glob
from gendre_classification import VAD, properties
from tqdm import tqdm
import librosa
import numpy as np
import pandas


librispeech_path = 'D:\\Università\\Tesi\\Datasets\\LibriSpeech\\dev-clean\\'
# audio_list = glob.glob(librispeech_path + "/*/*/*.flac")
# F_audio_list = glob.glob(librispeech_path + )
# M_audio_list =
txt_path = 'D:\\Università\\Tesi\\Datasets\\LibriSpeech\\SPEAKERS2.TXT'
data_info = pandas.read_fwf(txt_path, delimiter='\t|')
data_subset_info = data_info.loc[data_info[' SUBSET           '] == ' dev-clean        ']


male_subset = data_subset_info.loc[data_subset_info['SEX'] == ' M ']
female_subset = data_subset_info.loc[data_subset_info['SEX'] == ' F ']

male_ID = pandas.Series.tolist(male_subset['ID   '])
female_ID = pandas.Series.tolist(female_subset['ID   '])

# male audio list
male_audio_list = []
for i in male_ID:
    path = glob.glob(librispeech_path + str(i) + '/*/*.flac')
    male_audio_list.extend(path)
# print(data_info.head())
DATA= []
for i in range(len(ID)):
    id = ID[i]
    sex = SEX[i]
    data = [id, sex]
    DATA.append(data)
# Mettiamo a posto i dati


# data_subset_info = data_info.loc[data_info['SUBSET'] == 'dev-clean']
# print(data_subset_info.head())

percentage_data = []
offset_value_data = []
elaborated_audiofile_path_data = []

flag = 0

def audio_load(audiofile_path, attempts):
    x, fc = librosa.load(audiofile_path, sr=properties.fs)
    d = len(x) / properties.fs - properties.dur
    off_set = np.random.uniform(0, d)
    x, fc = librosa.load(audiofile_path, sr=properties.fs, duration=properties.dur, offset=off_set)
    percentuale = calc_percentage(VAD.run_vad(data=x))
    if (percentuale > 50):
        if (attempts <= properties.max_attempts):
            attempts += 1;
            return audio_load(audiofile_path, attempts)
        else:
            return x, percentuale, off_set

def calc_percentage(seg):
    i = 0  # counter per is_speech 'True'
    j = 0  # counter per is_speech 'False'
    for s in seg:
        for k in s:
            if (k == 'is_speech'):
                if (s[k] == True):
                    i = i + s['stop'] - s['start']
                else:
                    j = j + s['stop'] - s['start']
    percentuale = i / (i + j) * 100
    return percentuale

for audio_path in tqdm(audio_list):
    Y, percentuale, offset = audio_load(audio_path, 0)
    id = audio_path.split('\\')[6]
#     output_saving_path = output_path + 'speechfile_N_{}.npy'.format(flag)
#     np.save(output_saving_path, Y)
#     elaborated_audiofile_path_data.append(output_saving_path)
#     percentage_data.append(percentuale)
#     offset_value_data.append(offset)
    flag += 1
#
# # Dataframe creation
# df = pandas.DataFrame(
#     {'Elaborated audiofile path': elaborated_audiofile_path_data,
#      'Speech percentage': percentage_data,
#      'Offset Value': offset_value_data
#      }
# )
#
# print(df.head())
# df.to_csv(output_path + 'elaborated_audiofile_INFO.csv')
