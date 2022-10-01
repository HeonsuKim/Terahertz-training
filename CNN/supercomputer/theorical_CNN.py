from scipy import signal
from scipy import io
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import tensorflow as tf
#import h5py

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

#with h5py.File('D:\\Dropbox\\THz\\MATLAB\\deconvolution\\result_del.mat', 'r') as f:
#    print(list(f.keys()))
#    a = list(f['result_pulse'])

wb=openpyxl.load_workbook('C:\\Users\\user03\\PycharmProjects\\heonsu\\GFRP\\time.xlsx')
sheet= wb['Sheet1']
xx=sheet['A']
time=[]
for cell in xx:
    time.append(cell.value)

experiment=[]
label=[]
experiment_test=[]
label_test=[]

fs=18.10341 #THz
nperseg=8

a=io.loadmat('D:\\Dropbox\\THz\\MATLAB\\deconvolution\\finall2.mat')
#a.keys()

aa=a['aa']
signal_sur=a['signal_sur']
signal_bot=a['signal_bot']
signal_del=a['signal_del']
signal_noise=a['signal_noise']

for i in range(signal_sur.shape[1]):       #dat.shape[1]-1

    f, t, Zxx = signal.stft(signal_sur[:,i], fs=fs, nperseg=nperseg)
    Zxx = np.where(np.abs(Zxx) >= 0.02, Zxx, 0)
    t, xrec = signal.istft(Zxx, fs)
    signal_sur[:,i]=xrec[0:210].reshape(210)

    experiment.append(signal_sur[0+55:100+55,i])
    label_temp=[0 for j in range(110+4)]
    label_temp[111]=1
    label.append(label_temp)

print('signal_sur: complete')

for i in range(signal_bot.shape[1]):       #dat.shape[1]-1

    f, t, Zxx = signal.stft(signal_bot[:,i], fs=fs, nperseg=nperseg)
    Zxx = np.where(np.abs(Zxx) >= 0.02, Zxx, 0)
    t, xrec = signal.istft(Zxx, fs)
    signal_bot[:,i]=xrec[0:210].reshape(210)

    experiment.append(signal_bot[0+55:100+55,i])
    label_temp=[0 for j in range(110+4)]
    label_temp[112]=1
    label.append(label_temp)

print('signal_bot: complete')

for i in range(signal_noise.shape[1]):       #dat.shape[1]-1

    f, t, Zxx = signal.stft(signal_noise[:,i], fs=fs, nperseg=nperseg)
    Zxx = np.where(np.abs(Zxx) >= 0.02, Zxx, 0)
    t, xrec = signal.istft(Zxx, fs)
    signal_noise[:,i]=xrec[0:210].reshape(210)

    experiment.append(signal_noise[0+55:100+55,i])
    label_temp=[0 for j in range(110+4)]
    label_temp[0]=1
    label.append(label_temp)

print('signal_noise: complete')

for i in range(signal_del.shape[1]):       #dat.shape[1]-1

    f, t, Zxx = signal.stft(signal_del[:,i], fs=fs, nperseg=nperseg)
    Zxx = np.where(np.abs(Zxx) >= 0.02, Zxx, 0)
    t, xrec = signal.istft(Zxx, fs)
    signal_del[:,i]=xrec[0:210].reshape(210)


    if i%28365 < aa[0][0] and i%28365 >= 0 :
        experiment.append(signal_del[0+55:100+55,i])
        label_temp=[0 for j in range(110+4)]
        label_temp[1]=1
        label.append(label_temp)
    else:
        for j in range(110-1):

            aa_sum=0
            for jj in range(j+1):
                aa_sum=aa_sum+aa[0][jj]

            if i%28365 < aa_sum+aa[0][j+1] and i%28365 >= aa_sum :
                experiment.append(signal_del[0+55:100+55, i])
                label_temp = [0 for j in range(110 + 3)]
                label_temp[j+2] = 1
                label.append(label_temp)

print('signal_del: complete')

a=io.loadmat('D:\\Dropbox\\THz\\MATLAB\\deconvolution\\finall2_test.mat')
#a.keys()

aa=a['aa']
signal_sur=a['signal_sur']
signal_bot=a['signal_bot']
signal_del=a['signal_del']
signal_noise=a['signal_noise']

for i in range(signal_sur.shape[1]):       #dat.shape[1]-1

    f, t, Zxx = signal.stft(signal_sur[:,i], fs=fs, nperseg=nperseg)
    Zxx = np.where(np.abs(Zxx) >= 0.02, Zxx, 0)
    t, xrec = signal.istft(Zxx, fs)
    signal_sur[:,i]=xrec[0:210].reshape(210)

    experiment_test.append(signal_sur[0+55:100+55,i])
    label_temp=[0 for j in range(110+4)]
    label_temp[111]=1
    label_test.append(label_temp)

print('signal_sur: complete')

for i in range(signal_bot.shape[1]):       #dat.shape[1]-1

    f, t, Zxx = signal.stft(signal_bot[:,i], fs=fs, nperseg=nperseg)
    Zxx = np.where(np.abs(Zxx) >= 0.02, Zxx, 0)
    t, xrec = signal.istft(Zxx, fs)
    signal_bot[:,i]=xrec[0:210].reshape(210)

    experiment_test.append(signal_bot[0+55:100+55,i])
    label_temp=[0 for j in range(110+4)]
    label_temp[112]=1
    label_test.append(label_temp)

print('signal_bot: complete')

for i in range(signal_noise.shape[1]):       #dat.shape[1]-1

    f, t, Zxx = signal.stft(signal_noise[:,i], fs=fs, nperseg=nperseg)
    Zxx = np.where(np.abs(Zxx) >= 0.02, Zxx, 0)
    t, xrec = signal.istft(Zxx, fs)
    signal_noise[:,i]=xrec[0:210].reshape(210)

    experiment_test.append(signal_noise[0+55:100+55,i])
    label_temp=[0 for j in range(110+4)]
    label_temp[0]=1
    label_test.append(label_temp)

print('signal_noise: complete')

for i in range(signal_del.shape[1]):       #dat.shape[1]-1

    f, t, Zxx = signal.stft(signal_del[:,i], fs=fs, nperseg=nperseg)
    Zxx = np.where(np.abs(Zxx) >= 0.02, Zxx, 0)
    t, xrec = signal.istft(Zxx, fs)
    signal_del[:,i]=xrec[0:210].reshape(210)


    if i%28365 < aa[0][0] and i%28365 >= 0 :
        experiment_test.append(signal_del[0+55:100+55,i])
        label_temp=[0 for j in range(110+4)]
        label_temp[1]=1
        label_test.append(label_temp)
    else:
        for j in range(110-1):

            aa_sum=0
            for jj in range(j+1):
                aa_sum=aa_sum+aa[0][jj]

            if i%28365 < aa_sum+aa[0][j+1] and i%28365 >= aa_sum :
                experiment_test.append(signal_del[0+55:100+55, i])
                label_temp = [0 for j in range(110 + 3)]
                label_temp[j+2] = 1
                label_test.append(label_temp)

print('signal_del: complete')

fs=18.10341 #THz
nperseg=8
spectrogram=[]
spectrogram_test=[]
for i in range(len(experiment)):
    f, t, Zxx = signal.stft(experiment[i], fs=fs, nperseg=nperseg)
    spectrogram.append(Zxx)

for i in range(len(experiment_test)):
    f, t, Zxx = signal.stft(experiment_test[i], fs=fs, nperseg=nperseg)
    spectrogram_test.append(Zxx)



input_h = spectrogram[0].shape[0]
input_w = spectrogram[0].shape[1]
input_ch = 1

num_classes = 114

model = Sequential([
  layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(input_h, input_w, input_ch)),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  #layers.Dense(num_classes)

  layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              #loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


spectrogram_np=np.array(spectrogram).reshape((len(spectrogram),input_h,input_w,1))
spectrogram_test_np=np.array(spectrogram_test).reshape((len(spectrogram_test),input_h,input_w,1))

label_np=np.zeros(len(label))
for i in range(len(label)):
    label_np[i]=np.argmax(label[i])

label_test_np=np.zeros(len(label_test))
for i in range(len(label_test)):
    label_test_np[i]=np.argmax(label_test[i])

model.fit(spectrogram_np, label_np, epochs=15)

test_loss, test_acc = model.evaluate(spectrogram_test_np,  label_test_np, verbose=2)

predictions = model.predict(spectrogram_test_np)