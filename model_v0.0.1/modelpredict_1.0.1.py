# file to predict using the TP1.0 architecture
from dataprep import *
import keras
import sys

# helper function to intepret probabilities
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

# function to actually generate text
def generate_text(ex_path, master_path, model, text_len):
    index =1 #for shakes cleaning
    maxChar=100
    td = TextData(master_path, index, maxChar)   
    alphabet = td.alphabet
    int_to_char = td.int_to_char
    char_to_int = td.char_to_int

    # input is the cleaned text
    td_sample = TextData(ex_path, index, maxChar)
    input = td_sample.clean_text
    
    # make sure at least 40 characters for training
    if len(input) < maxChar:
        raise ValueError('Input must have >= %i characters. You have %i.' %(maxChar, len(input)))
    print('input:')
    print(input)
    print('-----------------')
    print('output:')
    # grab last maxChar characters
    sentence = input[-maxChar:]
    #sentence = input
    #print(sentence)

    # initalize generated string
    generated = ''
    # don't append input
    # generated += input
        
    # randomly pick diversity parameter
    diversities = [0.2, 0.5, 1.0, 1.2]
    div_index = int(np.random.random()*(len(diversities)))
    diversity = diversities[div_index]
    #print('diversity:', diversity)
    #sys.stdout.write(input)

    # generate text_len characters worth of test
    for i in range(text_len):
        # prepare chosen sentence as part of new dataset
        x_pred = np.zeros((1, len(sentence), len(alphabet)))
        for t, char in enumerate(sentence):
            x_pred[0, t, char_to_int[char]] = 1.0

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

    return generated

# function to actually generate text
def generate_text_text(text, master_path, model, text_len):
    index =1 #for shakes cleaning
    maxChar=100
    td = TextData(master_path, index, maxChar)   
    alphabet = td.alphabet
    int_to_char = td.int_to_char
    char_to_int = td.char_to_int

    # input is the cleaned text
    td_sample = TextDataText(text, index, maxChar)
    input = td_sample.clean_text
    
    # make sure at least 40 characters for training
    if len(input) < maxChar:
        raise ValueError('Input must have >= %i characters. You have %i.' %(maxChar, len(input)))
    print('input:')
    print(input)
    print('-----------------')
    print('output:')
    # grab last maxChar characters
    sentence = input[-maxChar:]
    #sentence = input
    #print(sentence)

    # initalize generated string
    generated = ''
    # don't append input
    # generated += input
        
    # randomly pick diversity parameter
    diversities = [0.2, 0.5, 1.0, 1.2]
    div_index = int(np.random.random()*(len(diversities)))
    diversity = diversities[div_index]
    #print('diversity:', diversity)
    #sys.stdout.write(input)

    # generate text_len characters worth of test
    for i in range(text_len):
        # prepare chosen sentence as part of new dataset
        x_pred = np.zeros((1, len(sentence), len(alphabet)))
        for t, char in enumerate(sentence):
            x_pred[0, t, char_to_int[char]] = 1.0

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

    return generated

# EX_PATH = '/home/oscar47/Desktop/thinking_parrot/input.txt'# path to input 
# MASTER_PATH = '/home/oscar47/Desktop/thinking_parrot/texts/master.txt'# path for the training text file
# model = keras.models.load_model('/home/oscar47/Desktop/thinking_parrot/Literary-RNN/model_v0.0.1/models/shakespeare_v0.0.1.hdf5')
# output_char_num = 400
# generate_text(EX_PATH, MASTER_PATH, model, output_char_num)
