# master file for Shakespeare hyperparam sweeps
# @oscars47

#imports

# defaults
import os
from numpy import random

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
import dataprep
from dataprep import *


# enter path to text here cleaned-------------------
#path = 'THE SONNETS.txt'
path = 'sonnets_mini.txt'

# prepare data for training--------------------
# set max_Char value; this is length of sentence which we train on
maxChar = 100
# index 1 for shakespeare cleaning
index = 1

# create TextData object for the Shakespeare text
td = dataprep.TextData(path, index, maxChar)
alphabet, char_to_int, int_to_char = td.get_parsed()
text = td.get_text()

# generate training set
x_train, y_train, x_val, y_val = td.prepare_data()

# build model functions--------------------------------
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
              config.dropout_1, config.dropout_2,  config.dropout_3, config.dropout_4, config.dropout_5, config.learning_rate)
      
      #now run training
      history = model.fit(
        x_train, y_train,
        batch_size = config.batch_size,
        validation_data=(x_val, y_val),
        epochs=config.epochs,
        callbacks=callbacks #use callbacks to have w&b log stats; will automatically save best model                     
      )  

def build_model(LSTM_layer_size_1,  LSTM_layer_size_2, LSTM_layer_size_3, 
          LSTM_layer_size_4, LSTM_layer_size_5, 
          dropout_1, dropout_2,  dropout_3, dropout_4, dropout_5, learning_rate):
    # call initialize function
    
    model = Sequential()
    # RNN layers for language processing
    model.add(LSTM(LSTM_layer_size_1, input_shape = (maxChar, len(alphabet)), return_sequences=True))
    model.add(Dropout(dropout_1))

    model.add(LSTM(LSTM_layer_size_2, return_sequences=True))
    model.add(Dropout(dropout_2))

    model.add(LSTM(LSTM_layer_size_3, return_sequences=True))
    model.add(Dropout(dropout_3))

    model.add(LSTM(LSTM_layer_size_4, return_sequences=True))
    model.add(Dropout(dropout_4))

    model.add(LSTM(LSTM_layer_size_5))
    model.add(Dropout(dropout_5))

    model.add(Dense(len(alphabet)))
    model.add(Activation('softmax'))


    # put structure together
    optimizer = RMSprop(learning_rate = learning_rate)
    model.compile(loss='categorical_crossentropy')

    return model

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
       'value': 5
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
    'dropout_1': {
      'distribution': 'uniform',
       'min': 0,
       'max': 0.6
    },
     'dropout_2': {
       'distribution': 'uniform',
       'min': 0,
       'max': 0.6
    },
     'dropout_3': {
             'distribution': 'uniform',
       'min': 0,
       'max': 0.6
    },
     'dropout_4': {
             'distribution': 'uniform',
       'min': 0,
       'max': 0.6
    },
     'dropout_5': {
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
wandb.init(project="Thinking-Parrot-ShakespeareTest", entity="oscarscholin")

# finish with callbacks------------
# use the two helper functions above to create the LambdaCallback 
print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

# define two other callbacks
# save model
# if no directory "models" exists, create it
if not(os.path.exists('models')):
    os.mkdir('./models/')
modelpath = "models/shakespeare_v0.0.1.hdf5"
checkpoint = ModelCheckpoint(modelpath, monitor='loss',
                             verbose=1, save_best_only=True,
                             mode='min')
# if learning stals, reduce the LR
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2,
                              patience=1, min_lr=0.001)

# compile the callbacks
callbacks = [print_callback, checkpoint, reduce_lr, WandbCallback()]

# initialize sweep!

sweep_id = wandb.sweep(sweep_config, project='Thinking-Parrot-ShakespeareTest', entity="oscarscholin")

# 'train' tells agent function is train
# 'count': number of times to run this
wandb.agent(sweep_id, train, count=100)