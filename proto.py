import numpy as np
# tarik salameh, prototype 0 nn for genomic classification

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d
from tflearn.layers.estimator import regression

import h5py
h = h5py.File('datasets/train.h5','r')
X, Y = h['X'],h['Y']
Y = np.reshape(Y, (-1, 1))
hh = h5py.File('datasets/val.h5','r')
Xv, Yv = hh['X'],hh['Y']
Yv = np.reshape(Yv, (-1, 1))

# Building 'AlexNet'
network = input_data(shape=[None, 6371, 38, 3])
network = conv_2d(network, 128, 38, strides=38, activation='relu')
network = dropout(network, 0.5)
network = conv_2d(network, 32, 38, activation='relu')
network = fully_connected(network,1,activation='linear')
network = regression(network, optimizer='adam',
                     loss='mean_square',
                     learning_rate=0.001)

# Training
model = tflearn.DNN(network, checkpoint_path='model_proto',
                    max_checkpoints=1, tensorboard_verbose=3)
model.fit(X, Y, n_epoch=3, validation_set=(Xv,Yv),
          show_metric=True, batch_size=38, snapshot_step=1,
          snapshot_epoch=False, run_id='ms_proto')
ht = h5py.File('datasets/test.h5','r')
Xt, Yt = ht['X'],ht['Y']
Yt = np.reshape(Yt, (-1, 1))
model.predict(Xt)

