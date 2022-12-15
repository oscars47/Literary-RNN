# file to train network
# @oscars47

import os
import numpy as np
from keras.callbacks import LambdaCallback, ModelCheckpoint, ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Activation
from keras.optimizers import RMSprop
import tensorflow as tf
import wandb
from wandb.keras import *
import sys

# check GPU num
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

from dataprep2 import TextData # import TextData class for processing
from modelpredict2 import * # get functions to interpret output

# define path
MAIN_DIR = '/home/oscar47/Desktop/thinking_parrot'
DATA_DIR = os.path.join(MAIN_DIR, 'texts_prep') # main
# DATA_DIR = os.path.join(MAIN_DIR, 'texts_prep', 'test') # for testing

# define master txt file
MASTER_TEXT_PATH = os.path.join(MAIN_DIR, 'texts', 'master.txt')
#MASTER_TEXT_PATH = os.path.join(MAIN_DIR, 'texts', 'toaster_man.txt')

# initialize text object
maxChar = 50
master=TextData(MASTER_TEXT_PATH, maxChar)
# get alphabet
alphabet = master.alphabet
char_to_int= master.char_to_int
int_to_char = master.int_to_char
text = master.text

# read in files for training
x_train = np.load(os.path.join(DATA_DIR, 'x_train.npy'))
y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
x_val = np.load(os.path.join(DATA_DIR, 'x_val.npy'))
y_val = np.load(os.path.join(DATA_DIR, 'y_val.npy'))

# build model functions--------------------------------
def build_model(LSTM_layer_size_1,  LSTM_layer_size_2, LSTM_layer_size_3, 
          LSTM_layer_size_4, LSTM_layer_size_5, 
          dropout, learning_rate):
    # call initialize function
    
    model = Sequential()
    # RNN layers for language processing
    model.add(LSTM(LSTM_layer_size_1, input_shape = (2*maxChar, len(alphabet)), return_sequences=True))
    model.add(LSTM(LSTM_layer_size_2, return_sequences=True))
    model.add(LSTM(LSTM_layer_size_3, return_sequences=True))
    model.add(LSTM(LSTM_layer_size_4, return_sequences=True))
    model.add(LSTM(LSTM_layer_size_5))

    model.add(Dropout(dropout))

    model.add(Dense(len(alphabet)))
    model.add(Activation('softmax'))


    # put structure together
    optimizer = RMSprop(learning_rate = learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy')

    return model

def train(config=None):
    with wandb.init(config=config):
    # If called by wandb.agent, as below,
    # this config will be set by Sweep Controller
      config = wandb.config

      #pprint.pprint(config)

      #initialize the neural net; 
      global model
      model = build_model(config.LSTM_layer_size_1,  config.LSTM_layer_size_2, config.LSTM_layer_size_3, 
              config.LSTM_layer_size_4, config.LSTM_layer_size_5, 
              config.dropout, config.learning_rate)
      
      #now run training
      history = model.fit(
        x_train, y_train,
        batch_size = config.batch_size,
        validation_data=(x_val, y_val),
        epochs=config.epochs,
        callbacks=callbacks #use callbacks to have w&b log stats; will automatically save best model                     
      ) 

def train_custom(LSTM_layer_size_1=128,  LSTM_layer_size_2=128, LSTM_layer_size_3=128, 
              LSTM_layer_size_4=128, LSTM_layer_size_5=128, 
              dropout=0.1, learning_rate=0.01, epochs=1, batchsize=32):
    #initialize the neural net; 
    global model
    model = build_model(LSTM_layer_size_1,  LSTM_layer_size_2, LSTM_layer_size_3, 
            LSTM_layer_size_4, LSTM_layer_size_5, 
            dropout, learning_rate)
    
    #now run training
    history = model.fit(
    x_train, y_train,
    batch_size = batchsize,
    validation_data=(x_val, y_val),
    epochs=epochs,
    callbacks=callbacks #use callbacks to have w&b log stats; will automatically save best model                     
    )

def train_custom_resume(model, batchsize, epochs):
    #now run training
    history = model.fit(
    x_train, y_train,
    batch_size = batchsize,
    validation_data=(x_val, y_val),
    epochs=epochs,
    callbacks=callbacks #use callbacks to have w&b log stats; will automatically save best model                     
    )

# helper functions from Keras

def get_toast_len(mean, stdev):
    toast_len = int(np.random.normal(mean, stdev))
    return toast_len

# do this each time we begin a new epoch    
def on_epoch_end(epoch, logs):
    # Function invoked at end of each epoch. Prints generated text.
    print()
    print('----- Generating text after Epoch: %d' % epoch)


    for diversity in [0.1, 0.5,1.2]:
        print('----- diversity:', diversity)

        start_index = np.random.randint(0, len(text) - maxChar - 1) +1
        # need to check how much to pad
        if start_index < maxChar:
            sentence0 = text[0:start_index] # up to but not including start index
            sentence1 = text[start_index+1: start_index+start_index+1]
            sentence = sentence0+sentence1
        else:
            stdev = (1/2)*(maxChar - 1)
            mean = (maxChar - 1)
                 # compute len, following normal distribution between 1 and maxChar; will go from [:num] as first part then [num+1:] concatenated; predict at num
                # need toastlen positive but no more than  maxChar
            goodtoast = False
            while goodtoast==False:
                toast_len = get_toast_len(mean, stdev)
                # add 1 to len since the distr here goes from 0 up to maxChar-1
                toast_len+=1
                #print(toast_len)
                if (toast_len > 0) and (toast_len <= maxChar): # if get acceptable toast, can leave
                    goodtoast=True
                    break
            sentence0 = text[start_index-toast_len:start_index]
            sentence1 = text[start_index+1: start_index+toast_len]
            sentence =  sentence0+ sentence1
        
        # need another condition here about if neat the end

        # 1. compute difference from maxChar and len/2
        diff = maxChar - int(len(sentence)/2)
        # need to check even/odd so we don;t overcount
        if len(sentence) %2 != 0: # if odd: need to subtract 1
            diff-=1

        # 2. initialize new string for each sentence
        complete_sentence = ''
        for i in range(diff):
            complete_sentence+='£' # appending forbidden
        # 3. now add 'real' sentence
        complete_sentence+=sentence
        # 4. append forbidden again
        for i in range(diff):
            complete_sentence+='£'

        print(len(complete_sentence))
        print(len(sentence))

        generated = ''
        #generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        #sys.stdout.write(generated)

        # generate 400 characters worth of test
        for i in range(400):
            # prepare chosen sentence as part of new dataset
            x_pred = np.zeros((1, 2*maxChar, len(alphabet)))
            #x_pred = np.zeros((2*maxChar, len(alphabet)))
            for t, char in enumerate(complete_sentence):
                if char != '£': # encode 1 iff it's not padded
                    x_pred[0, t, char_to_int[char]] = 1.
                    #x_pred[t, char_to_int[char]] = 1.

            # use the current model to predict what outputs are
            preds = model.predict(x_pred, verbose=0)[0] # removed [0] here
            # call the function above to interpret the probabilities and add a degree of freedom
            next_index = sample(preds, diversity)
            #convert predicted number to character
            next_char = int_to_char[next_index]

            generated+=next_char

            # check size of sentence; if still small can keep old stuff in sentence0
            if len(sentence) >= 2*maxChar:
                sentence0 = sentence0[1:]
            sentence0 += next_char # append new middle character
            sentence=sentence0+sentence1 # append to main sentence

            # print the new character as we create it
            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()

# define search parameters-----------------
# holds wandb config nested dictionaries
# @oscars47

# set dictionary with random search; optimizing val_loss
sweep_config= {
    'method': 'random',
    'name': 'val_loss',
    'goal': 'minimize'
}

sweep_config['metric']= 'val_loss'

# now name hyperparameters with nested dictionary
parameters_dict = {
    'epochs': {
       'value':5
    },
    # for build_dataset
     'batch_size': {
       'distribution': 'int_uniform',  #we want to specify a distribution type to more efficiently iterate through these hyperparams
       'min': 64,
       'max': 128
    },
    'LSTM_layer_size_1': {
       'distribution': 'int_uniform',
       'min': 64,
       'max': 256
    },
    'LSTM_layer_size_2': {
       'distribution': 'int_uniform',
       'min': 64,
       'max': 256
    },
    'LSTM_layer_size_3': {
       'distribution': 'int_uniform',
       'min': 64,
       'max': 256
    },
    'LSTM_layer_size_4': {
       'distribution': 'int_uniform',
       'min': 64,
       'max': 256
    },
    'LSTM_layer_size_5': {
       'distribution': 'int_uniform',
       'min': 64,
       'max': 256
    },
     'dropout': {
             'distribution': 'uniform',
       'min': 0,
       'max': 0.6
    },
    'learning_rate':{
         #uniform distribution between 0 and 1
         'distribution': 'uniform', 
         'min': 0,
         'max': 0.1
     }
}

# append parameters to sweep config
sweep_config['parameters'] = parameters_dict

# login to wandb-------------------------
#wandb.init(project="Thinking-Parrot2.0-1", entity="oscarscholin")

# finish with callbacks------------
# use the two helper functions above to create the LambdaCallback 
print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

# define two other callbacks
# save model
# if no directory "models" exists, create it
if not(os.path.exists(os.path.join(MAIN_DIR, 'models'))):
    os.mkdir(os.path.join(MAIN_DIR, 'models'))
modelpath = os.path.join(MAIN_DIR, "tp2_v0.0.1.hdf5")
checkpoint = ModelCheckpoint(modelpath, monitor='loss',
                             verbose=1, save_best_only=True,
                             mode='min')
# if learning stals, reduce the LR
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2,
                              patience=1, min_lr=0.001)

# compile the callbacks
#callbacks = [print_callback, checkpoint, reduce_lr, WandbCallback()]
callbacks = [print_callback, checkpoint, reduce_lr]

# initialize sweep!

# sweep_id = wandb.sweep(sweep_config, project="Thinking-Parrot2.0-1", entity="oscarscholin")

# # 'train' tells agent function is train
# # 'count': number of times to run this
# wandb.agent(sweep_id, train, count=100)

# train_custom(LSTM_layer_size_1=248,  LSTM_layer_size_2=194, LSTM_layer_size_3=210, 
#               LSTM_layer_size_4=122, LSTM_layer_size_5=256, 
#               dropout=0.1, learning_rate=0.01, epochs=25, batchsize=96)

# continue training------
MODEL_PATH = os.path.join(MAIN_DIR, 'tp2_v0.0.1.hdf5')
model = keras.models.load_model(MODEL_PATH)
train_custom_resume(model, 96, 25)