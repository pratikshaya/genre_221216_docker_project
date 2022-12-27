
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import librosa as lb
import pandas as pd
import matplotlib.pyplot as plt

from keras.layers.pooling import GlobalMaxPooling2D
from keras.models import Model
from keras import optimizers 
from keras import backend as K
from keras.layers import Input, Dense
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
#from keras.layers.advanced_activations import  ReLU
from keras.layers.advanced_activations import ELU

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

sr=16000
N_FFT      = 512
N_MELS     = 96
N_OVERLAP  = 256
DURA       = 30
Class_num = 5
     
def PreTrained_CNN(weights='msd', input_tensor=None,include_top=True):
    if weights not in {'msd', None}:
        raise ValueError('The `weights` is either `None` or `msd` Million Song Dataset')

    # Determine proper input shape
    if K.image_dim_ordering() == 'th':
        input_shape = (1, 96, 1876)  
    else:
        input_shape = (96, 1876, 1)  #10 second audio length and sr (16k=626 and 12k = 469)

    if input_tensor is None:
        melgram_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            melgram_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            melgram_input = input_tensor

    # Determine input axis
    if K.image_dim_ordering() == 'th':
        channel_axis = 1
        freq_axis = 2
        time_axis = 3
    else:
        channel_axis = 3
        freq_axis = 1
        time_axis = 2

    # Input block
    x = ZeroPadding2D(padding=(0, 37))(melgram_input)
    x = BatchNormalization(axis=time_axis, name='bn_0_freq')(x)

    # Conv block 1
    x = Convolution2D(32, 3, 3, border_mode='same', name='conv2d_1')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='batch_normalization_1')(x)
    x = ELU()(x)
    #x = ReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2), name='MP_1')(x)

    # Conv block 2
    x = Convolution2D(32, 3, 3, border_mode='same', name='conv2d_2')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='batch_normalization_2')(x)
    x = ELU()(x)
    #x = ReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2), name='MP_2')(x)

    # Conv block 3
    x = Convolution2D(32, 3, 3, border_mode='same', name='conv2d_3')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='batch_normalization_3')(x)
    x = ELU()(x)
    #x = ReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2), name='MP_3')(x)
    
    # Conv block 4
    x = Convolution2D(32, 3, 3, border_mode='same', name='conv2d_4')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='batch_normalization_4')(x)
    x = ELU()(x) 
    #x = ReLU()(x) #Check both bottomneck feature and Fine tuning
    x = MaxPooling2D(pool_size=(2, 2), name='MP_4')(x)
    
    # Conv block 5
    x = Convolution2D(32, 3, 3, border_mode='same', name='conv2d_5')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='batch_normalization_5')(x)
    x = ELU()(x)
    #x = ReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2), name='MP_5')(x)
    x = GlobalMaxPooling2D(name='MP')(x)
    #x = GlobalAveragePooling2D(name='GAP')(x)
    
    if include_top:
        x = Dense(50, activation='sigmoid')(x)

    # Create model
    model = Model(melgram_input, x)
    if weights is None:
        return model
    else:
        weights_path = '/home/bhubon/fine_tuning_genre/genre_221216_docker_project/model_best.hdf5'
        model.load_weights(weights_path, by_name=True)

        return model


#Audio path
#path = 'D:/EmotionDB/AUDIO/'
path = '/home/bhubon/fine_tuning_genre/genre_221216_docker_project/dataset/'
labels_file  = '/home/bhubon/fine_tuning_genre/genre_221216_docker_project/data_csv/genre_221216.csv'
#tags =['Exciting', 'Fear', 'Neutral', 'Relaxation','Sad', 'Tension']
tags = ['Court_music_C', 'Creative_gugak_R', 'Folk_music_F', 'Fusion_gugak_S', 'Pungryu_music_E']
labels = pd.read_csv(labels_file,header=0)

def log_scale_melspectrogram(path):
    #print("what is path",path)
    #print("sr",sr)
    signal, sr_n = lb.load(path, sr=sr)
    n_sample = signal.shape[0]
    n_sample_fit = int(DURA*sr_n)
    
    if n_sample < n_sample_fit:
        signal = np.hstack((signal, np.zeros((int(DURA*sr_n) - n_sample,))))
    elif n_sample > n_sample_fit:
        signal = signal[int((n_sample-n_sample_fit)/2):int((n_sample+n_sample_fit)/2)]
        
    melspect = lb.core.amplitude_to_db(lb.feature.melspectrogram(y=signal, sr=sr_n, hop_length=N_OVERLAP, n_fft=N_FFT, n_mels=N_MELS)**2)
    
    return melspect 


def get_labels(labels_dense=labels['label']):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * Class_num
    labels_one_hot = np.zeros((num_labels, Class_num))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

def get_melspectrograms(labels_dense=labels):
    spectrograms = np.asarray([log_scale_melspectrogram(i) for i in labels_dense['filepath']])
    spectrograms = spectrograms.reshape(spectrograms.shape[0], spectrograms.shape[1], spectrograms.shape[2], 1)
    return spectrograms
        
def get_melspectrograms_indexed(index, labels_dense=labels):
    spectrograms = np.asarray([log_scale_melspectrogram(i) for i in labels_dense['filepath'][index]])
    print('Spectrogram shape:', spectrograms.shape)
    spectrograms = spectrograms.reshape(spectrograms.shape[0], spectrograms.shape[1], spectrograms.shape[2], 1)
    return spectrograms


if __name__ == '__main__':
    n_samples = 3014 # Total data 2848
    #cv_split      = 0.95   #0.8                             
    #train_size    = int(n_samples * cv_split)                               
    #test_size     = int(n_samples - train_size)
    
    #Loading data
    indices = np.arange(n_samples)  #n_samples =2961
    #np.random.shuffle(indices)
    train_indices = indices[0:n_samples]
    #test_indices  = indices[train_size:]
    
    labels = get_labels(labels_dense=labels['label'])
    X_train = get_melspectrograms_indexed(train_indices)



    y_train = labels[train_indices]
       
    #from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import KFold
    
    seed = 7
    np.random.seed(seed)
    X, Y = X_train, y_train

    #kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
    kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
    print(kfold)
    cvscores = []
    
    for train, test in kfold.split(X, Y):
        print('loading the model and the pre-trained weights...')
        base_model = PreTrained_CNN(weights='msd', include_top=False)
        x = base_model.output
        print("The model value", x.shape)
        out = Dense(Class_num,activation='softmax', name = "Output_Dense")(x)
    
        model = Model(input=base_model.input, output=out)
    
        model.compile(loss="categorical_crossentropy", optimizer=optimizers.RMSprop(lr=0.0001, rho=0.9),metrics=["accuracy"])
        print(model.summary())
    
        hist = model.fit(X[train], Y[train], batch_size= 10, nb_epoch=40)
        
        (loss, accuracy) = model.evaluate(X[test], Y[test], batch_size=32, verbose=1)
        
        scores = model.predict(X[test])
        
        y_pred = np.argmax(scores, axis=1)
        y_test = np.argmax(Y[test], axis=1)
        
        labels = ['Court_music_C', 'Creative_gugak_R', 'Folk_music_F', 'Fusion_gugak_S', 'Pungryu_music_E']
        
        cm = confusion_matrix(y_test, y_pred)
        
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        
        disp.plot(cmap=plt.cm.Blues)
        plt.show()
        
        cvscores.append(accuracy)
        
print('the accuracy for each fold',cvscores)
print("the mean of 5 fold cross validation is", np.mean(cvscores))
        
