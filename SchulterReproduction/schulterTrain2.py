try:
    from code.schultercore5 import *
except ImportError:
    from schultercore5 import *   # when running from terminal, the directory may not be identified as a package
from keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import pickle
import matplotlib.pyplot as plt
import sys
import time

start_time=time.time()
print('Ignore this test print' +str(time.time()-start_time))

params=load_parameters()

#just to get number of files per set, used further down in step computing
print('gathering files...')
train_dir = 'jamendo/audioTrain/'
train_files = [train_dir + x for x in os.listdir(train_dir) if x.endswith('.wav') or x.endswith('.ogg') or x.endswith('.mp3')]
val_dir = 'jamendo/audioVal/'
val_files = [val_dir + x for x in os.listdir(val_dir) if x.endswith('.wav') or x.endswith('.ogg') or x.endswith('.mp3')]
test_dir = 'jamendo/audioTest/'
test_files = [test_dir + x for x in os.listdir(test_dir) if x.endswith('.wav') or x.endswith('.ogg') or x.endswith('.mp3')]
print(len(train_files), 'train files \n', len(val_files), 'validation files \n', len(test_files), 'test files')


# make sure we have the hdf5 data file
hdf5_path = 'hdf5data/' +sys.argv[1] +'.hdf5'
if not os.path.isfile:
    print('ERROR: HDF5-file not found! Run create_hdf5_dataset.py first!')
    exit(0)

# load parameters
params = load_parameters()

# generate CNN model
model = generate_network(params)
model.summary()  # print a summary of the model

# pre-compute number of steps
hdf5_file = h5py.File(hdf5_path, "r")
# num_val_steps = params['num_train_steps'] * len(val_files)/len(train_files)
num_val_steps=0
for i in range(params['songs_to_validate']):
	num_val_steps = num_val_steps + int(hdf5_file['val_lengths'][i])/params['batch_size']

num_val_steps=int(num_val_steps)
# callbacks
# save the best performing model
save_best = ModelCheckpoint('models/' +sys.argv[2] +'.h5', monitor='val_loss', save_best_only=True)
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto', baseline=None, restore_best_weights=False)

print(params)

# train
history = model.fit_generator(train_generator(params['num_train_steps'], hdf5_path, params),
                    steps_per_epoch=params['num_train_steps'],
                    epochs=params['epochs'],
                    validation_data=val_generator(num_val_steps, hdf5_path, params),
                    validation_steps=num_val_steps,
                    callbacks=[save_best,early_stop])
print(params)
print('Training took: ' +str(time.time()-start_time))

# save model history to disk as the same name as model
pickle_out = open('modelHistory/' +sys.argv[2] +'history.pickle', 'wb')
pickle.dump(history.history, pickle_out)
pickle_out.close()
