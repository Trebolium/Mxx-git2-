try:
    from code.schultercore import *
except ImportError:
    from schultercore import *   # when running from terminal, the directory may not be identified as a packagefrom keras.callbacks import ModelCheckpoint
# schultercore.py has additional functions
import os, shutil, librosa
import numpy as np
import time
import matplotlib.pyplot as plt
import h5py
import csv
import math

params=load_parameters()

print('gathering files...')
train_dir = 'jamendo/audioTrain/'
train_files = [train_dir + x for x in os.listdir(train_dir) if x.endswith('.wav') or x.endswith('.ogg') or x.endswith('.mp3')]
val_dir = 'jamendo/audioVal/'
val_files = [val_dir + x for x in os.listdir(val_dir) if x.endswith('.wav') or x.endswith('.ogg') or x.endswith('.mp3')]
test_dir = 'jamendo/audioTest/'
test_files = [test_dir + x for x in os.listdir(test_dir) if x.endswith('.wav') or x.endswith('.ogg') or x.endswith('.mp3')]
print(len(train_files), 'train files \n', len(val_files), 'validation files \n', len(test_files), 'test files')

test_folder_dir= 'jamendo/testFolder/'
test_folder_files = [test_folder_dir + x for x in os.listdir(test_folder_dir) if x.endswith('.wav') or x.endswith('.ogg') or x.endswith('.mp3')]

print('setting up hdf5 file...')

num_train_instances = len(train_files)
num_val_instances = len(val_files)

image_height=params['n_mel']
image_width=int(np.round(params['max_song_length'] * params['fs'] / float(params['hop_length'])))

dataset = h5py.File('jamendo/jdataset.hdf5', mode='w')

dataset.create_dataset('train_labels',
                       shape=(num_train_instances, 1),
                       dtype=np.int)
dataset.create_dataset('val_labels',
                       shape=(num_val_instances, 1),
                       dtype=np.int)
# we store one image per instance of size image_height x image_width holding floating point numbers
dataset.create_dataset('train_features',
                       shape=(num_train_instances, image_height, image_width+1),
                       dtype=np.float)
dataset.create_dataset('val_features',
                       shape=(num_val_instances, image_height, image_width+1),
                       dtype=np.float)

# TRAIN TRAIN SET
print('working on train set...')
name_length_list=[]

for k, audio_path in enumerate(train_files):
    print('  ', k + 1, '/', len(train_files), audio_path)
    # extract the feature
    feature, audio_melframe_nums = extract_feature(audio_path, params)
    name_length_list.append([os.path.basename(audio_path),audio_melframe_nums])
    plot_save_feature(feature, os.path.basename(audio_path))
    # we can deduce the label from the file name
    dataset['train_features'][k, ...] = feature
    
with open('jamendo/trainSongLength.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(name_length_list)
csvFile.close()

# TRAIN VAL SET
print('working on validation set...')
name_length_list=[]

for k, audio_path in enumerate(val_files):
    print('  ', k + 1, '/', len(val_files), audio_path)
    # extract the feature
    feature, audio_melframe_nums = extract_feature(audio_path)
    name_length_list.append([os.path.basename(audio_path),audio_melframe_nums])
    plot_save_feature(feature, os.path.basename(audio_path))
    # we can deduce the label from the file name
    dataset['val_features'][k, ...] = feature
    
with open('jamendo/valSongLength.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(name_length_list)
csvFile.close()