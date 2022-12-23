# master file for Shakespeare hyperparam sweeps
# @oscars47

#imports

# defaults
import os
from numpy import random
import sys

# machine learning libraries
from keras.callbacks import LambdaCallback, ModelCheckpoint, ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
import keras, pprint # pprint allows us to visualize nested dictionaries

# hyperparameter optmization and visualization
import wandb
from wandb.keras import WandbCallback
from modelpredict import *

#import other files
from dataprepsn import TextData, TextDataText


# enter path to text here cleaned-------------------
MAIN_DIR = '/home/oscar47/Desktop/thinking_parrot'
TEXT_DIR= os.path.join(MAIN_DIR, 'texts','master.txt')
DATA_DIR=os.path.join(MAIN_DIR, 'texts_0.0.2_sn_prep')
MODEL_PATH = os.path.join(MAIN_DIR, 'tp-1.2 models')

# load in data
x_train = np.load(os.path.join(DATA_DIR, 'x_train.npy'))
y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
x_val = np.load(os.path.join(DATA_DIR, 'x_val.npy'))
y_val = np.load(os.path.join(DATA_DIR, 'y_val.npy'))

# prepare data for training--------------------
# set max_Char value; this is length of sentence which we train on
maxChar = 100
# index 1 for shakespeare cleaning
index = 1
master=TextData(TEXT_DIR, index, maxChar)
# get alphabet
alphabet = master.alphabet
char_to_int= master.char_to_int
int_to_char = master.int_to_char
text = master.clean_text

# create TextData object for the Shakespeare text


# build model functions--------------------------------
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

def build_model(LSTM_layer_size_1,  LSTM_layer_size_2, LSTM_layer_size_3, 
          LSTM_layer_size_4, LSTM_layer_size_5, 
          dropout, learning_rate):
    # call initialize function
    
    model = Sequential()
    # RNN layers for language processing
    model.add(LSTM(LSTM_layer_size_1, input_shape = (maxChar, len(alphabet)), return_sequences=True))
    model.add(LSTM(LSTM_layer_size_2, return_sequences=True))
    model.add(LSTM(LSTM_layer_size_3, return_sequences=True))
    model.add(LSTM(LSTM_layer_size_4, return_sequences=True))
    model.add(LSTM(LSTM_layer_size_5))
    model.add(Dropout(dropout))

    model.add(Dense(len(alphabet)))
    model.add(Activation('softmax'))


    # put structure together
    optimizer = RMSprop(learning_rate = learning_rate)
    model.compile(loss='categorical_crossentropy')

    return model

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

# do this each time we begin a new epoch    
def on_epoch_end(epoch, logs):
    # Function invoked at end of each epoch. Prints generated text.
    print()
    print('----- Generating text after Epoch: %d' % epoch)

    start_index = random.randint(0, len(text) - maxChar - 1)
    for diversity in [0.1, 0.2, 0.5, 1.0, 1.2, 1.5, 2.0]:
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + maxChar]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        #sys.stdout.write(generated)

        # generate 400 characters worth of test
        for i in range(400):
            # prepare chosen sentence as part of new dataset
            x_pred = np.zeros((1, maxChar, len(alphabet)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_to_int[char]] = 1.

            # use the current model to predict what outputs are
            preds = model.predict(x_pred, verbose=0)[0]
            # call the function above to interpret the probabilities and add a degree of freedom
            next_index = sample(preds, diversity)
            #convert predicted number to character
            next_char = int_to_char[next_index]

            # append to existing string so as to build it up
            generated += next_char
            # append new character to previous sentence and delete the old one in front; now we train on predictions
            sentence = sentence[1:] + next_char

            # print the new character as we create it
            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()


# login to wandb-------------------------
#wandb.init(project="Thinking-Parrot-ShakespeareTest", entity="oscarscholin")

# finish with callbacks------------
# use the two helper functions above to create the LambdaCallback 
print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

# define two other callbacks
# save model
# if no directory "models" exists, create it
if not(os.path.exists('models2')):
    os.mkdir('./models2/')
modelpath = "models2/shakespeare_v0.0.1.hdf5"
checkpoint = ModelCheckpoint(modelpath, monitor='loss',
                             verbose=1, save_best_only=True,
                             mode='min')
# if learning stals, reduce the LR
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2,
                              patience=1, min_lr=0.001)

# compile the callbacks
#callbacks = [print_callback, checkpoint, reduce_lr, WandbCallback()]
callbacks = [print_callback, checkpoint, reduce_lr]

# custom training!-----
# train_custom(LSTM_layer_size_1=248,  LSTM_layer_size_2=194, LSTM_layer_size_3=210, 
#               LSTM_layer_size_4=122, LSTM_layer_size_5=256, 
#                dropout=0.1, learning_rate=0.01, epochs=25, batchsize=96)

# resume training!
# MAIN_DIR = '/home/oscar47/Desktop/thinking_parrot'
# MODEL_PATH = os.path.join(MAIN_DIR, 'Literary-RNN/model_v0.0.1/models/shakespeare_v0.0.1.hdf5')
model = keras.models.load_model(modelpath)
train_custom_resume(model, 96, 25)