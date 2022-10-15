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

#training dataset
spectrogram_np=np.load('D:\\Dropbox\\python_data\\data\\spectrogram_np.npy')
label_np=np.load('D:\\Dropbox\\python_data\\data\\label_np.npy')

#test dataset
spectrogram_test_np=np.load('D:\\Dropbox\\python_data\\data\\spectrogram_np_test.npy')
label_test_np=np.load('D:\\Dropbox\\python_data\\data\\label_np_test.npy')

#Convolutional Neural Network (CNN)
input_h = 5
input_w = 26
input_ch = 1
num_classes = 104

model = Sequential([
  layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(input_h, input_w, input_ch)),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes, activation='softmax')])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#CNN training
hist = model.fit(spectrogram_np, label_np, epochs=30, validation_data=(spectrogram_test_np, label_test_np))

model.save('D:\\Dropbox\\python_data\\data\\my_model.h5')
with open('D:\\Dropbox\\python_data\\data\\hist.pkl', 'wb') as pw:
    pickle.dump(hist.history, pw)


#Training curve
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()