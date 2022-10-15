from scipy import signal
from scipy import io
import matplotlib.pyplot as plt
import pickle
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

gpus=tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu,True)
        logical_gpus=tf.config.experimental.list_physical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

experiment=[]
label=[]
experiment_test=[]
label_test=[]

fs=18.10341 #THz
nperseg=8

#import THz training dataset and labelling
a=io.loadmat('D:\\Dropbox\\python_data\\data\\data11.mat')
a2=io.loadmat('D:\\Dropbox\\python_data\\data\\data13.mat')

aa=a['aa']
signal_sur1=(a['signal_sur']*100).astype(np.int8)
signal_sur=(a2['signal_peak']*100).astype(np.int8)
signal_del=(a['signal_del']*100).astype(np.int8)
signal_noise=(a['signal_noise']*100).astype(np.int8)

#THz single signal
for i in range(30):

    f, t, Zxx = signal.stft(signal_sur1[:,25650*i+25649], fs=fs, nperseg=nperseg)
    Zxx = np.where(np.abs(Zxx) >= 0.02, Zxx, 0)
    t, xrec = signal.istft(Zxx, fs)
    signal_sur1[:,25650*i+25649]=xrec[0:210].reshape(210)

    experiment.append(signal_sur1[0+55+30:100+55-10,25650*i+25649])
    label_temp=[0 for j in range(100+4)]
    label_temp[101]=1
    label.append(label_temp)


for i in range(int(signal_sur.shape[1])):

    f, t, Zxx = signal.stft(signal_sur[:,i], fs=fs, nperseg=nperseg)
    Zxx = np.where(np.abs(Zxx) >= 0.02, Zxx, 0)
    t, xrec = signal.istft(Zxx, fs)
    signal_sur[:,i]=xrec[0:210].reshape(210)

    experiment.append(signal_sur[0+55+30:100+55-10,i])
    label_temp=[0 for j in range(100+4)]
    label_temp[101]=1
    label.append(label_temp)

print('signal_sur: complete')

#THz noise signal
for i in range(30):

    f, t, Zxx = signal.stft(signal_noise[:,i], fs=fs, nperseg=nperseg)
    Zxx = np.where(np.abs(Zxx) >= 0.02, Zxx, 0)
    t, xrec = signal.istft(Zxx, fs)
    signal_noise[:,i]=xrec[0:210].reshape(210)

    experiment.append(signal_noise[0+55+30:100+55-10,i])
    label_temp=[0 for j in range(100+4)]
    label_temp[0]=1
    label.append(label_temp)

print('signal_noise: complete')

#THz overlapped signal
for i in range(signal_del.shape[1]):

    f, t, Zxx = signal.stft(signal_del[:,i], fs=fs, nperseg=nperseg)
    Zxx = np.where(np.abs(Zxx) >= 0.02, Zxx, 0)
    t, xrec = signal.istft(Zxx, fs)
    signal_del[:,i]=xrec[0:210].reshape(210)


    if i%25650 < aa[0][0]-102 and i%25650 >= 0+102 :
        experiment.append(signal_del[0+55+30:100+55-10,i])
        label_temp=[0 for j in range(100+4)]
        label_temp[1]=1
        label.append(label_temp)
    else:
        for j in range(100-1):

            aa_sum=0
            for jj in range(j+1):
                aa_sum=aa_sum+aa[0][jj]

            if i%25650 < aa_sum+aa[0][j+1]-102 and i%25650 >= aa_sum+102 :
                experiment.append(signal_del[0+55+30:100+55-10, i])
                label_temp = [0 for j in range(100+4)]
                label_temp[j+2] = 1
                label.append(label_temp)

print('signal_del: complete')

#THz trucated signal
for i in range(30):

    f, t, Zxx = signal.stft(signal_sur[:,30*i], fs=fs, nperseg=nperseg)
    Zxx = np.where(np.abs(Zxx) >= 0.02, Zxx, 0)
    t, xrec = signal.istft(Zxx, fs)
    signal_sur[:,30*i]=xrec[0:210].reshape(210)

    for k in range(50):
        experiment.append(signal_sur[30+k:90+k,30*i])
        label_temp=[0 for j in range(100+4)]
        label_temp[103]=1
        label.append(label_temp)
        experiment.append(signal_sur[140-k:200-k, 30*i])
        label_temp=[0 for j in range(100+4)]
        label_temp[103]=1
        label.append(label_temp)

print('other_sur: complete')


for i in range(30):

    f, t, Zxx = signal.stft(signal_noise[:,25650*i+12825], fs=fs, nperseg=nperseg)
    Zxx = np.where(np.abs(Zxx) >= 0.02, Zxx, 0)
    t, xrec = signal.istft(Zxx, fs)
    signal_noise[:,25650*i+12825]=xrec[0:210].reshape(210)

    for k in range(50):
        experiment.append(signal_noise[30+k:90+k,25650*i+12825])
        label_temp=[0 for j in range(100+4)]
        label_temp[103]=1
        label.append(label_temp)
        experiment.append(signal_noise[140-k:200-k, 25650*i+12825])
        label_temp=[0 for j in range(100+4)]
        label_temp[103]=1
        label.append(label_temp)

print('other_noise: complete')


for i in range(signal_del.shape[1]):

    f, t, Zxx = signal.stft(signal_del[:,i], fs=fs, nperseg=nperseg)
    Zxx = np.where(np.abs(Zxx) >= 0.02, Zxx, 0)
    t, xrec = signal.istft(Zxx, fs)
    signal_del[:,i]=xrec[0:210].reshape(210)


    if i%25650 < aa[0][0] and i%25650 >= 0 :
        for k in range(50):
            experiment.append(signal_del[0+k:100+k,i])
            label_temp = [0 for j in range(100 + 4)]
            label_temp[103] = 1
            label.append(label_temp)
            experiment.append(signal_del[0+55+55-k:100+55+55-k, i])
            label_temp=[0 for j in range(100+4)]
            label_temp[103]=1
            label.append(label_temp)
    else:
        for j in range(100-1):
            aa_sum=0
            for jj in range(j+1):
                aa_sum=aa_sum+aa[0][jj]

            if i%25650 < aa_sum+aa[0][j+1] and i%25650 >= aa_sum :
                for k in range(50):
                    experiment.append(signal_del[0+k:100+k, i])
                    label_temp = [0 for j in range(100 + 4)]
                    label_temp[103] = 1
                    label.append(label_temp)
                    experiment.append(signal_del[0+55+55-k:100+55+55-k, i])
                    label_temp = [0 for j in range(100+4)]
                    label_temp[103] = 1
                    label.append(label_temp)

print('other_del: complete')


#import THz testing dataset and labelling
a=io.loadmat('D:\\Dropbox\\python_data\\data\\data12.mat')
a2=io.loadmat('D:\\Dropbox\\python_data\\data\\data14.mat')

aa=a['aa']
signal_del=a['signal_del']
signal_noise=a['signal_noise']
signal_sur=a2['signal_peak']

#THz single signal
for i in range(5):

    f, t, Zxx = signal.stft(signal_sur[:,25650*i+25649], fs=fs, nperseg=nperseg)
    Zxx = np.where(np.abs(Zxx) >= 0.02, Zxx, 0)
    t, xrec = signal.istft(Zxx, fs)
    signal_sur[:,25650*i+25649]=xrec[0:210].reshape(210)

    experiment_test.append(signal_sur[0+55:100+55,25650*i+25649])
    label_temp=[0 for j in range(100+4)]
    label_temp[101]=1
    label_test.append(label_temp)

print('signal_sur: complete')

#THz noise signal
idx = np.random.randint(int(signal_noise.shape[1]),size=int(signal_noise.shape[1]/25650))
for i in range(5):

    f, t, Zxx = signal.stft(signal_noise[:,25650*i+1], fs=fs, nperseg=nperseg)
    Zxx = np.where(np.abs(Zxx) >= 0.02, Zxx, 0)
    t, xrec = signal.istft(Zxx, fs)
    signal_noise[:,25650*i+1]=xrec[0:210].reshape(210)

    experiment_test.append(signal_noise[0+55:100+55,25650*i+1])
    label_temp=[0 for j in range(100+4)]
    label_temp[0]=1
    label_test.append(label_temp)

print('signal_noise: complete')

#THz overlapped signal
for i in range(signal_del.shape[1]):

    f, t, Zxx = signal.stft(signal_del[:,i], fs=fs, nperseg=nperseg)
    Zxx = np.where(np.abs(Zxx) >= 0.02, Zxx, 0)
    t, xrec = signal.istft(Zxx, fs)
    signal_del[:,i]=xrec[0:210].reshape(210)


    if i%25650 < aa[0][0] and i%25650 >= 0 :
        experiment_test.append(signal_del[0+55:100+55,i])
        label_temp=[0 for j in range(100+4)]
        label_temp[1]=1
        label_test.append(label_temp)
    else:
        for j in range(100-1):

            aa_sum=0
            for jj in range(j+1):
                aa_sum=aa_sum+aa[0][jj]

            if i%25650 < aa_sum+aa[0][j+1] and i%25650 >= aa_sum :
                experiment_test.append(signal_del[0+55:100+55, i])
                label_temp = [0 for j in range(100+4)]
                label_temp[j+2] = 1
                label_test.append(label_temp)

print('signal_del: complete')


#Short-term Fourier transform
fs=18.10341 #THz
nperseg=8

spectrogram=[]
spectrogram_test=[]
for i in range(len(experiment)):
    f, t, Zxx = signal.stft(experiment[i], fs=fs, nperseg=nperseg)
    spectrogram.append(Zxx.astype(np.int8))
    print("process: "+str(i/len(experiment)*100)+"\n")

for i in range(len(experiment_test)):
    f, t, Zxx = signal.stft(experiment_test[i], fs=fs, nperseg=nperseg)
    spectrogram_test.append(Zxx.astype(np.int8))
    print("process: "+str(i/len(experiment_test)*100)+"\n")


spectrogram_np=np.zeros((len(spectrogram),input_h,input_w,1)).astype(np.int8)
for i in range(len(spectrogram)):
    spectrogram_np[i]=spectrogram[i].reshape((input_h,input_w,1))

spectrogram_test_np=np.zeros((len(spectrogram_test),input_h,input_w,1))
for i in range(len(spectrogram_test)):
    spectrogram_test_np[i]=spectrogram_test[i].reshape((input_h,input_w,1))

label_np=np.zeros(len(label))
for i in range(len(label)):
    label_np[i]=np.argmax(label[i])

label_test_np=np.zeros(len(label_test))
for i in range(len(label_test)):
    label_test_np[i]=np.argmax(label_test[i])