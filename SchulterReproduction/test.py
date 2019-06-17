import pickle

pickle_in=open('modelHistory/deletehistory.pickle','rb')
example_dict=pickle.load(pickle_in)

print(example_dict['loss'])