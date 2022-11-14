# builds RNN model
# @oscars47

import os

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
import keras, pprint # pprint allows us to visualize nested dictionaries

import wandb
from dataprep import *
from configinfo import *

# extract maxChar and alphabet by passing td object
def initialize_vars(td, x_train, y_train, x_val, y_val): 
    global maxChar, alphabet, char_to_int, int_to_char, text, x_train, y_train, x_val, y_val, callbacks 
    maxChar = td.get_maxChar()
    alphabet, char_to_int, int_to_char = td.get_parsed()
    text = td.get_text()

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
              config.dropout_1, config.dropout_2,  config.dropout_3, config.dropout_4, config.dropout_5, config.learning_rate, td)
      
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
          dropout_1, dropout_2,  dropout_3, dropout_4, dropout_5, learning_rate, td):
    # call initialize function
    initialize_vars(td)
    
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

# interpret probabilities
def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    # rescale data
    preds = np.asarray(preds).astype('float64')
    #preds = np.log(preds) / temperature
    exp_preds = np.exp(1/temperature)*preds
    preds = exp_preds / np.sum(exp_preds)
    # create multinomial distribution; run experiment 10 times, select most probable outcome
    probas = np.random.multinomial(10, preds, 1)
    return np.argmax(probas)

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