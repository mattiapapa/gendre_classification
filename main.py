import glob
from tqdm import tqdm
import librosa
import numpy as np
import pandas
from gendre_classification import VAD
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

fs = 16000  # LibriSpeech sampling frequency [Hz]
dur = 1  # Audio duration [s]
win_dur = 0.02
max_attempts = 20

librispeech_path = 'D:\\Università\\Tesi\\Datasets\\LibriSpeech\\dev-clean\\'

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

# female audio list
female_audio_list = []
for i in female_ID:
    path = glob.glob(librispeech_path + str(i) + '/*/*.flac')
    female_audio_list.extend(path)


def audio_load(audiofile_path, attempts):
    x, fc = librosa.load(audiofile_path, sr=fs)
    d = len(x) / fs - dur
    off_set = np.random.uniform(0, d)
    x, fc = librosa.load(audiofile_path, sr=fs, duration=dur, offset=off_set)
    percentuale = calc_percentage(VAD.run_vad(data=x))
    if (percentuale < 50):
        if (attempts <= max_attempts):
            attempts += 1
            return audio_load(audiofile_path, attempts)
    else:
        return x, off_set

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


offset_list = []
path_list = []
feature_list = []
sex_list = []
for audio_path in tqdm(female_audio_list):
    Y, offset = audio_load(audio_path, 0)
    S_mel = librosa.feature.melspectrogram(Y, fs)
    offset_list.append(offset)
    path_list.append(audio_path)
    sex_list.append('F')
    feature_list.append(S_mel)
for audio_path in tqdm(male_audio_list):
    Y, offset = audio_load(audio_path, 0)
    S_mel = librosa.feature.melspectrogram(Y, fs)
    offset_list.append(offset)
    path_list.append(audio_path)
    sex_list.append('M')
    feature_list.append(S_mel)

df = pandas.DataFrame({
    'audio_path': path_list,
    'sex': sex_list,
    'features': feature_list,
    'offset': offset_list
})
# Random shuffle of data
shuffled_df = shuffle(df, random_state = 0) # random_state = 0 for reproducible results across multiple function calls
X = np.array(pandas.Series.tolist(shuffled_df['features']))
y = pandas.Series.tolist(shuffled_df['sex'])
X_train, X_test, y_train, y_test = train_test_split(X, y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train)

# Naive model: DecisionTreeClassifier
model_DecisionTree = DecisionTreeClassifier()
# it requires an input shape such as (n_data, n_features)
a = X_train.reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2])
b = X_test.reshape(X_test.shape[0], X_test.shape[1]*X_test.shape[2])
history = model_DecisionTree.fit(a, y_train)
p_train = model_DecisionTree.predict(a)
p_test = model_DecisionTree.predict(b)

from sklearn.metrics import confusion_matrix

import seaborn
data_train = confusion_matrix(y_train, p_train, labels=['M','F'], normalize='all')
data_test  = confusion_matrix(y_test,  p_test,  labels=['M','F'], normalize='all')
df_cm = pandas.DataFrame(data_train, columns=np.unique(y_train), index = np.unique(y_train))
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
plt.figure(figsize = (10,7))
seaborn.set(font_scale=1.4)#for label size
seaborn.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 12})# font size
plt.show()
df_cm_test = pandas.DataFrame(data_test, columns=np.unique(y_test), index = np.unique(y_test))
df_cm_test.index.name = 'Actual'
df_cm_test.columns.name = 'Predicted'
plt.figure(figsize = (10,7))
seaborn.set(font_scale=1.4)#for label size
seaborn.heatmap(df_cm_test, cmap="Blues", annot=True,annot_kws={"size": 12})# font size
plt.show()

print(f'Accuracy on Train: {df_cm["F"][0]+df_cm["M"][1]} \n'
      f'Accuracy on Test:  {df_cm_test["F"][0]+df_cm_test["M"][1]}' )

from sklearn.preprocessing import LabelBinarizer
# F -> 0
# M -> 1
lb = LabelBinarizer()
labels_train = lb.fit_transform(np.array(y_train))
labels_val   = lb.fit_transform(np.array(y_val))
labels_test  = lb.fit_transform(np.array(y_test))

# Classification with GaussianNB
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
model_LR = GaussianNB()
model_LR.fit(a, labels_train)
p_train_LR = model_LR.predict(a)
p_test_LR = model_LR.predict(b)
acc_train = accuracy_score(labels_train, p_train_LR)
acc_test  = accuracy_score(labels_test, p_test_LR)

from sklearn.neural_network import MLPClassifier
model_MLPC = MLPClassifier(hidden_layer_sizes=[10,10])
model_MLPC.fit(a, labels_train)
p_train_MLPC = model_MLPC.predict(a)
p_test_MLPC = model_MLPC.predict(b)

acc_train_MLPC = accuracy_score(labels_train, p_train_MLPC)
acc_test_MLPC  = accuracy_score(labels_test, p_test_MLPC)


# CNN
import tensorflow.keras as keras

X_train = X_train[..., np.newaxis]
X_val   = X_val[..., np.newaxis]
X_test  = X_test[..., np.newaxis]

def build_model(input_shape):

    model = keras.Sequential()

    #input
    model.add(keras.layers.Conv2D(64,
                                  (3,3),
                                  activation='relu',
                                  padding='same',
                                  input_shape=input_shape))
    model.add(keras.layers.MaxPooling2D((2,2), strides=(2,2), padding='same'))

    model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))

    model.add(keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(4096, activation='relu'))
    model.add(keras.layers.Dense(4096, activation='relu'))

    # output
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    return model

def predict(model, X, y):
    X = X[np.newaxis, ...]
    prediction = model.predict(X)   # X -> [1,...,...,1]
    predicted_index = np.argmax(prediction, axis=1) # max on the axis 1: [] 1d array
    print(f'Expected index: {y}, Predicted index: {predicted_index}')


def plot_history(history):
    fig, axs = plt.subplots(2)
    # accuracy
    axs[0].plot(history.history['accuracy'], label='train accuracy')
    axs[0].plot(history.history['val_accuracy'], label='test accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].legend(loc='lower right')
    #axs[0].set_title('Accuracy eval [% error]')
    axs[0].grid(True)
    # error
    axs[1].plot(history.history['loss'], label='train error')
    axs[1].plot(history.history['val_loss'], label='test error')
    axs[1].set_ylabel('Error')
    axs[1].set_xlabel('Epoch')
    axs[1].legend(loc='upper right')
    #axs[1].set_title('Error eval')
    axs[1].grid(True)

    plt.show()


input_shape = (X_train.shape[1], # X_train.shape[0] would be the list
               X_train.shape[2],
               X_train.shape[3])

model_CNN = build_model(input_shape)

# compile the network

# optimizer = keras.optimizers.Adam(learning_rate=0.001)
# opt = keras.optimizers.SGD(lr=0.01, momentum=0.9)

# optimizer = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)

optimizer = keras.optimizers.Adam(learning_rate=0.001)
model_CNN.compile(optimizer=optimizer,
                  loss = 'binary_crossentropy',
                  metrics = ['accuracy']
                  )

# callbacks definition
weights_path = 'D:\\models_weights\\gendre_classification\\model_check_point'
callbacks = [keras.callbacks.EarlyStopping(patience = 10),
             keras.callbacks.ModelCheckpoint(filepath = weights_path,
                                             save_best_only = True,
                                             save_weights_only = True,
                                             monitor = 'val_loss'),
             keras.callbacks.ReduceLROnPlateau(patience=20)
             ]

# train the CNN
history = model_CNN.fit(X_train, labels_train,
                        validation_data = (X_val, labels_val),
                        batch_size = 32,
                        epochs = 50,
                        callbacks=callbacks)

# Plot History
plot_history(history)

# Evaluation on the test set
model_CNN.load_weights(weights_path)
print('Best weights loaded')
test_error, test_accuracy = model_CNN.evaluate(X_test, labels_test, verbose=1)
print(f'Accuracy on the test set is {test_accuracy}')

# Store all the predictions in Train&Test and plot Heatmap
for d in range(X_train):
    x = X_train[d]
    y = labels_train[d]
X_sample = X_test[89]
y_sample = labels_test[89]
predict(model_CNN, X_sample, y_sample)