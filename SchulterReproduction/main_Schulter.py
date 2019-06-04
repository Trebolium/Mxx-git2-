import os, shutil, librosa
import numpy as np
import keras

src_path = '/Users/brendanoconnor/Desktop/APP/MXX-git-/jamendo/audio'
dst_path = '/Users/brendanoconnor/Desktop/APP/MXX-git-/jamendo/audioConverted'
track_path= '/Users/brendanoconnor/Desktop/APP/MXX-git-/jamendo/01 - A better Life.ogg'

if not os.path.isdir(src_path):
    print('ERROR! source directory does not exist!: ', src_path)
if not os.path.isdir(dst_path):
    print('ERROR! Destination directory does not exist!: ', dst_path)
if not os.path.isfile(track_path):
    print('ERROR! File doesnt exist: ', track_path)

# Testing with single audio source
audio, sr = librosa.load(track_path)
D = np.abs(librosa.stft(audio))
print(D.shape)
D

mel_filter_bank=librosa.filters.mel(sr=22050,n_fft=1024,n_mels=80,fmin=27.5,fmax=8000,norm=1)
melfb_norm = (mel_filter_bank - np.mean(mel_filter_bank))/np.std(mel_filter_bank)



# Build directory structure

# data coming from
original_dataset_dir='/Users/brendanoconnor/Desktop/APP/MXX-git-/jamendo/audioConverted'
# data being arranged into
prepped_dataset_dir='/Users/brendanoconnor/Desktop/APP/MXX-git-/jamendo/audioPrepped'
os.mkdir(prepped_dataset_dir)

# Directories for our training,
# validation and test splits
train_dir = os.path.join(prepped_dataset_dir, 'train')
os.mkdir(train_dir)
validation_dir = os.path.join(prepped_dataset_dir, 'validation')
os.mkdir(validation_dir)
test_dir = os.path.join(prepped_dataset_dir, 'test')
os.mkdir(test_dir)

# ^ To be completed ^

# Build the model
from keras import layers
from keras import models

model = models.Sequential()
# ***change inputs to match melspec tensors***
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((3, 3)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((3, 3)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

from keras import optimizers

model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

# Implementation of Data Augmentation to be discussed
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=100,
      validation_data=validation_generator,
      validation_steps=50)