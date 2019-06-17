import pickle
import sys
import matplotlib.pyplot as plt

pickle_in=open('modelHistory/' +sys.argv[1] +'.pickle','rb')
model_history=pickle.load(pickle_in)

acc = model_history['acc']
val_acc = model_history['val_acc']
loss = model_history['loss']
val_loss = model_history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Model: ' +sys.argv[1] +', Training and validation loss')
plt.legend()

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Model: ' +sys.argv[1] +', Training and validation loss')
plt.legend()
plt.show()