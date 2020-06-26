import librosa
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import sounddevice as sd
from scipy.io.wavfile import write


fs = 44100  # Sample rate
seconds = 3 #length of the recording

def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
        return result

#Emotions in the RAVDESS dataset
emotions={
  '01':'neutral',
  '02':'calm',
  '03':'happy',
  '04':'sad',
  '05':'angry',
  '06':'fearful',
  '07':'disgust',
  '08':'surprised'
}
#Emotions to observe
observed_emotions=['calm', 'happy', 'fearful', 'disgust']

#Load the data and extract features for each sound file
def load_data(test_size=0.2):
    x,y=[],[]
    for file in glob.glob("C:\\Users\\Aditi\\Downloads\\speech_emotion_recognition\\ser_dataset\\Actor_*\\*.wav"):
        file_name=os.path.basename(file)
        emotion=emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)

#Split the dataset
x_train,x_test,y_train,y_test=load_data(test_size=0.25)

#Get the shape of the training and testing datasets
print((x_train.shape, x_test.shape))

#Get the number of features extracted
print(f'Features extracted: {x_train.shape[1]}')


#Initialize the Multi Layer Perceptron Classifier
model=MLPClassifier(alpha=0.001, batch_size= 350, epsilon=1e-08, hidden_layer_sizes=(355,), learning_rate='adaptive', max_iter=1000,random_state=0)

#Train the model
model.fit(x_train,y_train)

#Predict for the test set
y_pred=model.predict(x_test)

#Calculate the accuracy of our model
accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)
#Print the accuracy
print("Accuracy: {:.2f}%".format(accuracy*100))



#Record your own audio for testing the model
print("Now to test the model speak")
myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
sd.wait()  # Wait until recording is finished
write('output.wav', fs, myrecording)  # Save as WAV file
test=extract_feature('output.wav', mfcc=True, chroma=True, mel=True)


#Predict for the test input
test=test.reshape(1,-1)
y_pred=model.predict(test)
print("The emotion is : ")
print(y_pred)
