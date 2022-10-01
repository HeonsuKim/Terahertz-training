import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

wb=openpyxl.load_workbook('D:\\Dropbox\\THz\\data\\year2020\\20200922_gfrp\\wave.xlsx')
sheet= wb['Sheet1']
xx=sheet['A']
THz_Amp=[]
for cell in xx:
    THz_Amp.append(cell.value)

THz_Amp=np.array(THz_Amp)


lines = list()

with open('D:\\Dropbox\\THz\\data\\year2020\\20200922_gfrp\\del25_1_20200922150937_noise.txt', 'r', encoding='UTF-8') as file:
    for line in file:
        lines.append(line.strip('\n').split('\t'))


THz_Amp=np.zeros(1995)
THz_time=np.zeros(1995)
for i in range(1995):
    THz_Amp[i]=lines[i+19][2]
    THz_time[i]=lines[i+19][1]

fs=18.10341 #THz
nperseg=8

f, t, Zxx = signal.stft(THz_Amp, fs=fs, nperseg=nperseg)
#plt.pcolormesh(t, f, np.abs(Zxx), vmin=0)
Zxx = np.where(np.abs(Zxx) >= 15, Zxx, 0)
t, xrec = signal.istft(Zxx, fs)


THz_Amp= xrec[0:1995]
amp_max=max(THz_Amp)
amp_min=-min(THz_Amp)
for i in range(1995):
    if THz_Amp[i]>=0:
        THz_Amp[i]=(THz_Amp[i]/amp_max*100)
    else:
        THz_Amp[i]=(THz_Amp[i]/amp_min*100)

spectrogram_data=[]
for i in range(len(THz_Amp)-60):
    f, t, Zxx = signal.stft(THz_Amp[i:i+60], fs=fs, nperseg=nperseg)
    spectrogram_data.append(Zxx)

#spectrogram_data_np=np.array(spectrogram_data).reshape((len(spectrogram_data),spectrogram_data[0].shape[0],spectrogram_data[0].shape[1],1))
spectrogram_data_np=np.zeros((len(spectrogram_data),spectrogram_data[0].shape[0],spectrogram_data[0].shape[1],1))
for i in range(len(spectrogram_data)):
    spectrogram_data_np[i]=spectrogram_data[i].reshape((spectrogram_data[0].shape[0],spectrogram_data[0].shape[1],1))


predictions = model.predict(spectrogram_data_np)

for i in range(3):
    plt.figure()
    plt.plot(predictions.transpose()[i+101])
    plt.ylim([-0.1, 1.1])
plt.figure()
plt.plot(THz_Amp)



for i in range(104):
    plt.figure()
    plt.plot(predictions.transpose()[i])
    plt.ylim([-0.1, 1.1])
plt.figure()
plt.plot(THz_Amp)

np.savetxt('D:\\heonsu\\prediction_multi.txt', predictions)
np.savetxt('D:\\heonsu\\waveform_multi.txt', THz_Amp)

plt.close('all')