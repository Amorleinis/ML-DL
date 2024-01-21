import h5py
f = h5py.File('trained_model.h5', 'r')
list(f.keys())
['mydataset']
dset = f['mydataset']