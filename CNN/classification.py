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

#import actual THz signal
THz_signal=io.loadmat('D:\\Dropbox\\python_data\\data\\THz_signal.mat')

del25=(THz_signal['del25'][:,1]*100).astype(np.int8)
del110=(THz_signal['del110'][:,1]*100).astype(np.int8)
del275=(THz_signal['del275'][:,1]*100).astype(np.int8)
del450=(THz_signal['del450'][:,1]*100).astype(np.int8)
delmulti=(THz_signal['delmulti'][:,1]*100).astype(np.int8)

fs=18.10341 #THz
nperseg=8
input_h = 5#spectrogram[0].shape[0]
input_w = 26#spectrogram[0].shape[1]
input_ch = 1

f, t, Zxx = signal.stft(del25[:], fs=fs, nperseg=nperseg)
Zxx = np.where(np.abs(Zxx) >= 0.02, Zxx, 0)
t, xrec = signal.istft(Zxx, fs)
del25_denoised = xrec[:].reshape(1996)
valid_25 = []
for i in range(1995-100):
    valid_25.append(del25_denoised[i:100+i])
spectrogram_25 = []
for i in range(len(valid_25)):
    f, t, Zxx = signal.stft(valid_25[i], fs=fs, nperseg=nperseg)
    spectrogram_25.append(Zxx.astype(np.int8))
spectrogram_25_np=np.zeros((len(spectrogram_25),input_h,input_w,1)).astype(np.int8)
for i in range(len(spectrogram_25)):
    spectrogram_25_np[i]=spectrogram_25[i].reshape((input_h,input_w,1))


f, t, Zxx = signal.stft(del110[:], fs=fs, nperseg=nperseg)
Zxx = np.where(np.abs(Zxx) >= 0.02, Zxx, 0)
t, xrec = signal.istft(Zxx, fs)
del110_denoised = xrec[:].reshape(1996)
valid_110 = []
for i in range(1995-100):
    valid_110.append(del110_denoised[i:100+i])
spectrogram_110 = []
for i in range(len(valid_110)):
    f, t, Zxx = signal.stft(valid_110[i], fs=fs, nperseg=nperseg)
    spectrogram_110.append(Zxx.astype(np.int8))
spectrogram_110_np=np.zeros((len(spectrogram_110),input_h,input_w,1)).astype(np.int8)
for i in range(len(spectrogram_110)):
    spectrogram_110_np[i]=spectrogram_110[i].reshape((input_h,input_w,1))


f, t, Zxx = signal.stft(del275[:], fs=fs, nperseg=nperseg)
Zxx = np.where(np.abs(Zxx) >= 0.02, Zxx, 0)
t, xrec = signal.istft(Zxx, fs)
del275_denoised = xrec[:].reshape(1996)
valid_275 = []
for i in range(1995-100):
    valid_275.append(del275_denoised[i:100+i])
spectrogram_275 = []
for i in range(len(valid_275)):
    f, t, Zxx = signal.stft(valid_275[i], fs=fs, nperseg=nperseg)
    spectrogram_275.append(Zxx.astype(np.int8))
spectrogram_275_np=np.zeros((len(spectrogram_275),input_h,input_w,1)).astype(np.int8)
for i in range(len(spectrogram_275)):
    spectrogram_275_np[i]=spectrogram_275[i].reshape((input_h,input_w,1))


f, t, Zxx = signal.stft(del450[:], fs=fs, nperseg=nperseg)
Zxx = np.where(np.abs(Zxx) >= 0.02, Zxx, 0)
t, xrec = signal.istft(Zxx, fs)
del450_denoised = xrec[:].reshape(1996)
valid_450 = []
for i in range(1995-100):
    valid_450.append(del450_denoised[i:100+i])
spectrogram_450 = []
for i in range(len(valid_450)):
    f, t, Zxx = signal.stft(valid_450[i], fs=fs, nperseg=nperseg)
    spectrogram_450.append(Zxx.astype(np.int8))
spectrogram_450_np=np.zeros((len(spectrogram_450),input_h,input_w,1)).astype(np.int8)
for i in range(len(spectrogram_450)):
    spectrogram_450_np[i]=spectrogram_450[i].reshape((input_h,input_w,1))


f, t, Zxx = signal.stft(delmulti[:], fs=fs, nperseg=nperseg)
Zxx = np.where(np.abs(Zxx) >= 0.02, Zxx, 0)
t, xrec = signal.istft(Zxx, fs)
delmulti_denoised = xrec[:].reshape(1996)
valid_multi = []
for i in range(1995-100):
    valid_multi.append(delmulti_denoised[i:100+i])
spectrogram_multi = []
for i in range(len(valid_multi)):
    f, t, Zxx = signal.stft(valid_multi[i], fs=fs, nperseg=nperseg)
    spectrogram_multi.append(Zxx.astype(np.int8))
spectrogram_multi_np=np.zeros((len(spectrogram_multi),input_h,input_w,1)).astype(np.int8)
for i in range(len(spectrogram_multi)):
    spectrogram_multi_np[i]=spectrogram_multi[i].reshape((input_h,input_w,1))

#import CNN model
model = keras.models.load_model('D:\\Dropbox\\python_data\\data\\my_model.h5')
with open('D:\\Dropbox\\python_data\\data\\hist.pkl', 'rb') as pr:
    hist_load = pickle.load(pr)


#derive probability map
predictions_25 = model.predict(spectrogram_25_np)
predictions_110 = model.predict(spectrogram_110_np)
predictions_275 = model.predict(spectrogram_275_np)
predictions_450 = model.predict(spectrogram_450_np)
predictions_multi = model.predict(spectrogram_multi_np)

THz_signal=io.savemat('D:\\Dropbox\\python_data\\data\\predictions_25.mat', {'predictions_25':predictions_25})
THz_signal=io.savemat('D:\\Dropbox\\python_data\\data\\predictions_110.mat', {'predictions_110':predictions_110})
THz_signal=io.savemat('D:\\Dropbox\\python_data\\data\\predictions_275.mat', {'predictions_275':predictions_275})
THz_signal=io.savemat('D:\\Dropbox\\python_data\\data\\predictions_450.mat', {'predictions_450':predictions_450})
THz_signal=io.savemat('D:\\Dropbox\\python_data\\data\\predictions_multi.mat', {'predictions_multi':predictions_multi})