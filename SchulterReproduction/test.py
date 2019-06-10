import librosa
import yaml
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras import optimizers, models, layers
import h5py
import random
import os
import matplotlib.pyplot as plt
import math
import pdb

def load_parameters():
    return yaml.load(open('params.yaml'))

params=load_parameters()

hdf5_file=h5py.File('jamendo/jdataset.hdf5','r')

label_dataset='train_labels'
length_dataset='train_lengths'
feature_dataset='train_features'


x_data = []
y = []
# one loop per batch
for j in range(params['batch_size']):
	#random_song is the index of a random song entry
    random_song = random.randint(0,len(hdf5_file[label_dataset])-1)
    # = retrieves a feature from the shuffled list of songs
    feature = hdf5_file[feature_dataset][random_song, ...]
    # find how many samples are in this song by looking up lengths
    song_num_frames = hdf5_file[length_dataset][random_song, ...]
    #randomly take a section between 0 and the max available frame of a song which is as described below. Minusing one just 
    random_frame_index = random.randint(int(params['sample_frame_length']/2)+1  ,song_num_frames-int(params['sample_frame_length']/2)-1)
    # sample_excerpt must be a slice from the feature numpy
    sample_excerpt = feature[:,random_frame_index-int(params['sample_frame_length']/2)-1:random_frame_index+int(params['sample_frame_length']/2)]
    x_data.append(sample_excerpt)
    #convert samples into ms
    random_frame_time = random_frame_index*params['hop_length']/params['fs']
    #iterate through label_points rows until you find an entry number that is bigger than sample_excerpt_ms
    # if previous entry is even, there are vocals (1). Else no vocals (0)
    label_points=hdf5_file[label_dataset][random_song, ...]
    # determine via sample location whether window has vocals or not by comparing to csv
    previous_value=-1
    for row in range(500):
        # if the row is not the last (after last row value comes zero padding)
        if label_points[row][0]>previous_value:
            if label_points[row][0]>random_frame_time:
                # third element holds the label
                label=label_points[row-1][2]
                y.append(label)
                break
            else:
                previous_value=label_points[row][0]
        else:
            label=label_points[row][2]
            y.append(label)
            break
    print("song_index",random_song)
    print("song_num_frames", song_num_frames)
    print("random_frame_index", random_frame_index)
    print("random_frame_time", random_frame_time)
    # print("window_frame_boundaries",random_frame_index-int(params['sample_frame_length']/2)-1,random_frame_index+int(params['sample_frame_length']/2))
    # print("sample_excerpt.shape",sample_excerpt.shape)
    print("y", y)
    pdb.set_trace()
x_data = np.asarray(x_data)
# print(x_data.shape)
y = np.asarray(y)
# print(y.shape)
# conv layers need image data format
x_data = x_data.reshape((x_data.shape[0], x_data.shape[1], x_data.shape[2], 1))
# pdb.set_trace()
# send data at the end of each batch
