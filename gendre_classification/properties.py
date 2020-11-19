fs = 16000  # LibriSpeech sampling frequency [Hz]
dur = 1     # Audio duration [s]
win_dur = 0.02
max_attempts = 20

# librispeech_path = 'D:\\Università\\Tesi\\Datasets\\LibriSpeech\\dev-clean'
# audio_list = glob.glob(librispeech_path + "/*/*/*.flac")
#
# txt_path = 'D:\\Università\\Tesi\\Datasets\\LibriSpeech\\SPEAKERS2.TXT'
# data_info = pandas.read_fwf(txt_path, sep=' | ')
# print(data_info.head())
#
# data_subset_info = data_info.loc[data_info['SUBSET'] == 'dev-clean']
# print(data_subset_info.head())
#
# audio = audio_list[0]
# a, sr = librosa.load(audio, sr=fs)
# librosa.display.waveplot(a,sr)
# plt.show()
# S_mel = librosa.feature.melspectrogram(a, sr)
# print(a.shape)
# print(sr)
# print(S_mel.shape)
#
# # Audio graph
# plt.plot(a)
# plt.show()