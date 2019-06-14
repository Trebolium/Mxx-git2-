"""
STEP 2: Train a DNN to classify sounds as either 'cello' or 'applause'. This is a toy example and we are only training
for 5 epochs. Validation accuracy should go above 95%.
"""

try:
    from code.schultercore4 import *
except ImportError:
    from schultercore4 import *   # when running from terminal, the directory may not be identified as a package
from keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import matplotlib.pyplot as plt
import sys
import time

start_time=time.time()
print('Ignore this test print',time.time()-start_time)

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
total_training_examples = params['num_train_steps'] * params['batch_size']
num_val_steps = params['num_train_steps'] * len(val_files)/len(train_files)

# callbacks
# save the best performing model
save_best = ModelCheckpoint('models/' +sys.argv[2] +'.h5', monitor='val_loss', save_best_only=True)
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto', baseline=None, restore_best_weights=False)

print(params)

# train
history = model.fit_generator(data_generator('train', params['num_train_steps'], True, hdf5_path, params),
                    steps_per_epoch=params['num_train_steps'],
                    epochs=params['epochs'],
                    validation_data=data_generator('val', num_val_steps, False, hdf5_path, params),
                    validation_steps=num_val_steps,
                    callbacks=[save_best,early_stop])
print(params)
print(time.time()-start_time)

# create metrics for visualising
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Data: ' +sys.argv[1] +', Model: ' +sys.argv[2] +', Training and validation loss')
plt.legend()

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Data: ' +sys.argv[1] +', Model: ' +sys.argv[2] +', Training and validation loss')
plt.legend()
plt.show()