import librosa
import yaml
import numpy as np
import h5py
import random
import os
import matplotlib.pyplot as plt
import math
import pdb
import sys

data=h5py.File('jamendo/jdataset.hdf5','r')

label_dataset='train_labels'
length_dataset='train_lengths'
feature_dataset='train_features'

song=int(sys.argv[1])

for i in range(int(sys.argv[2])):
	print(data[label_dataset][song][i])
	