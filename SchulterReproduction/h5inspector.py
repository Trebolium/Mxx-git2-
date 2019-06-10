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

tlab='train_labels'
tlen='train_lengths'
tfea='train_features'

pdb.set_trace()

song=int(sys.argv[1])

for i in range(int(sys.argv[2])):
	print(data[tlab][song][i])
	