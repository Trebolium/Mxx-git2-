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


# UTILS
def load_parameters():
    """
    Load the parameter set from the yaml file and return as a python dictionary.

    Returns:
        dictionary holding parameter values
    """
    return yaml.load(open('params.yaml'))

# FEATURE EXTRACTION
def extract_feature(audio_path, params):
    audio, track_sr = librosa.load(audio_path, mono=True, sr=params['fs'])
    print('Track samplerate: ', track_sr)
    print('Track sample size: ', len(audio))
    
    # normalize
    audio /= max(abs(audio))
    max_samples = params['max_song_length'] * params['fs']  # desired max length in samples
    audio_melframe_nums = math.trunc(len(audio)/params['hop_length'])
    
    # either pad or cut to desired length
    if audio.shape[0] < max_samples:
        audio = np.pad(audio, (0, max_samples - audio.shape[0]), mode='constant')  # pad with zeros
    else:
        audio = audio[:max_samples]
    mel = librosa.feature.melspectrogram(audio,
                                             sr=params['fs'],
                                             n_mels=params['n_mel'],
                                             hop_length=params['hop_length'],
                                             n_fft=params['n_fft'], fmin=params['fmin'],fmax=params['fmax'])
    mel[mel < params['min_clip']] = params['min_clip']
    mel = librosa.amplitude_to_db(mel)
    mel = (mel - np.mean(mel))/np.std(mel)
    print(mel.shape)
    return mel, audio_melframe_nums

def plot_save_feature(dataset, feature, fname):
    fname = fname[:-4]
    plt.figure(figsize=(10, 4))
    plt.imshow(feature, aspect='auto', origin='lower')
    plt.title(fname)
    plt.colorbar()
    dataset = 'jamendo/image' +dataset +'/'
    plt.savefig(dataset +fname +'.png')
    plt.close(fname)

# NETWORK
def generate_network(params):
    """
    Generates a keras model with convolutional, pooling and dense layers.

    Args:
        parameters: Dictionary with system parameters.

    Returns:
        keras model object.
    """
    # pre-compute "image" height and width
    image_height = params['n_mel']
    # time to frames
    image_width = int(np.round(params['max_song_length'] * params['fs'] / float(params['hop_length'])))

    # standard
    model = models.Sequential()
    # ***change inputs to match melspec tensors***
    model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(image_height, params['sample_frame_length'], 1)))
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
    # model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))

    from keras import optimizers

    model.compile(loss='binary_crossentropy', optimizer=optimizers.SGD(lr=1e-2, momentum=0.95, decay=0.85, nesterov=True), metrics=['acc'])

    return model


# iterate through the hdf5 samples


    # for each randomly take a chunk of 115 consecutive frames




def data_generator(dataset, num_steps, shuffle, h5_path, params):
    """
    Data generator for training: Supplies the train method with features and labels taken from the hdf5 file

    Args:
        dataset: "train" or "test".
        num_steps: number of generation steps.
        shuffle: whether or not to shuffle the data
        h5_path: path to database .h5 file
        parameters: parameter dictionary

    Returns:
        feature data (x_data) and labels (y)
    """
    hdf5_file = h5py.File(h5_path, "r")  # open hdf5 file in read mode
    # point to the correct feature and label dataset
    feature_dataset = dataset + '_features'
    label_dataset = dataset + '_labels'
    length_dataset = dataset + '_lengths'
    while 1: # while 1 is the same is while True - an infinite loop!
        # for i in range(num_steps):  # one loop per epoch
        #     print("Numsteps: " +str(i+1) +"/" +str(num_steps))
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
            if sample_excerpt.shape[1]<params['sample_frame_length']:
                # print("batch number: ", j, "/", params['batch_size'])
                print("song_index",random_song)
                print("song_num_frames", song_num_frames)
                print("random_frame_index", random_frame_index)
                print("window_frame_boundaries",random_frame_index-int(params['sample_frame_length']/2)-1,random_frame_index+int(params['sample_frame_length']/2))
                print("sample_excerpt.shape",sample_excerpt.shape)
            x_data.append(sample_excerpt)
            #convert samples into ms
            random_frame_time = random_frame_index*params['hop_length']/params['fs']
            #iterate through label_points rows until you find an entry number that is bigger than sample_excerpt_ms
            # if previous entry is even, there are vocals (1). Else no vocals (0)
            label_points=hdf5_file[label_dataset][random_song, ...]
            # determine via sample location whether window has vocals or not by comparing to csv
            for row in range(500):
            	if label_points.shape[row]>random_frame_time:
            		if (row-1%2)==0:
            			label=1
            		else:
            			label=0
            		y.append(label)
            		break
        # print shape of x_data
        x_data = np.asarray(x_data)
        y = np.asarray(y)
        # conv layers need image data format
        x_data = x_data.reshape((x_data.shape[0], x_data.shape[1], x_data.shape[2], 1))
        # pdb.set_trace()
        # send data at the end of each batch
        yield x_data, y
