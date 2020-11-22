gendre_classification

Overview: 
- Data loading
- Data picking and labels association through the speaker.txt file
- Data processing and elaboration
- Classificators definition and performances evaluation 

LibriSpeech datasets: clean voices recorded, sampled @16Khz and ordered with an ID. The speaker.txt file collects all the IDs and provides detailed information
about the recordings, such as speaker name, sex, total duration of the recordings and subset they belong to. 

For the gender identification purpose, sex is labelled as the wanted ouput, both as a numerical value (i.e. 0 or 1) and categorical variable. 

PROCESSING STEP
Audiofiles are divided according to speakers gender. For each audio track, a total duration of 1 sec is chosen (that sampled at 16Khz, leads us to a vector of 16000 samples)
in which at least 50% of the samples contain speech activity (verified through a Voice Activity Detector from webrtcvad). Once the audiofiles have satisfied these requirements, 
they are loaded thorugh librosa and MFCCs are computed as features with default parameters (n_fft=2048, hop_length=512). 
The applifaction of librosa.feature.melspectrogram to an audiofile of duration 1 s, sampled at 16000 Hz, with default parameters, leads to a MFCCs of the form (128, 32). 

CLASSIFIERS 
GaussianNB, DecisionTreeClassifier and MLP Classifiers are chosen and implemented with sklearn library. Each classifier is trained and evaluated on the extracted MFCCs 
features and performances are reported with a confusion matrix, where in x_axis are reported the true values, in the y_axis the predicted ones. The diagonal of the 
confusion matrix tells us when the classifiers performs a correct prediction. 

At the end a CNN is implemented thorugh tensorflow.keras. The architecture of the model follows the famous VGGISH, but weights are not initialized and the all training 
is performed from random weights. From the model.sumary() it is possibile to see 84,858,113 trainable parameters. 
Before the evaluation on the test set, best weights stored thorugh the callbacks are loaded and then performances are monitored. In this case, all the metrics on train, validation 
and test are reported and confusion matrices produced. 
