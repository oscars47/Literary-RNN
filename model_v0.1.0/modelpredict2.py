# file to handle the prediction based on a trained model
# @oscars47

from dataprep2_uniform import TextData

import keras
import numpy as np
import sys

# helper function that we call to generate text
# takes in an input string, hdf5 trained model, and desired output length of text

def initialize_pre(td):
    global maxChar, alphabet, char_to_int, int_to_char, text 
    maxChar = td.get_maxChar()
    alphabet, char_to_int, int_to_char = td.get_parsed()
    text = td.get_text()
    print(alphabet)

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

def generate_text(ex_path, master_path, model, text_len):
    maxChar=50
    td = TextData(master_path, maxChar)   
    alphabet = td.alphabet
    int_to_char = td.int_to_char
    char_to_int = td.char_to_int

    # input is the cleaned text
    td_sample = TextData(ex_path, maxChar)
    input = td_sample.text[-2*maxChar+1:]
    print(len(input))

    
    # make sure at least 3 characters for training
    # if len(input) < 2*maxChar:
    #     raise ValueError('Input must have >= %i characters. You have %i.' %(3, len(input)))
    
    # grab last maxChar characters
    #sentence = input[-maxChar:]
    sentence = input
    # get sentence0 and sentence1
    sentence0 = sentence[:int(len(sentence)/2)]
    sentence1 = sentence[int(len(sentence)/2):]

    #print(sentence)

    # initalize generated string
    generated = ''
    # don't append input
    # generated += input
        
    # randomly pick diversity parameter
    diversities = [0.1, 0.2, 0.5, 1.0, 1.2, 1.5, 2.0]
    div_index = int(np.random.random()*(len(diversities)))
    diversity = diversities[div_index]
    #print('diversity:', diversity)
    #sys.stdout.write(input)

    # # 1. compute difference from maxChar and len/2
    # diff = maxChar - int(len(sentence)/2)
    # # 2. initialize new string for each sentence
    # complete_sentence = ''
    # for i in range(diff):
    #     complete_sentence+='£' # appending forbidden
    # # 3. now add 'real' sentence
    # complete_sentence+=sentence
    # # 4. append forbidden again
    # for i in range(diff):
    #     complete_sentence+='£'

    # generate text_len characters worth of test
    for i in range(text_len):
        # prepare chosen sentence as part of new dataset
        x_pred = np.zeros((1, 2*maxChar, len(alphabet)))
        for t, char in enumerate(sentence):
            if char != '£': # encode 1 iff it's not padded
                x_pred[0, t, char_to_int[char]] = 1.

        # use the current model to predict what outputs are
        preds = model.predict(x_pred, verbose=0)[0]
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

    return generated

# # function to call generate model >1 times
def experiment_RNN(input, model, text_len, num, maxChar, alphabet, char_to_int, int_to_char):
    for i in range(0, num):
        generate_text(input, model, text_len, maxChar, alphabet, char_to_int, int_to_char)
        print()

EX_PATH = '/home/oscar47/Desktop/thinking_parrot/input.txt'# path to input 
MASTER_PATH = '/home/oscar47/Desktop/thinking_parrot/texts/master.txt'# path for the training text file
model = keras.models.load_model('/home/oscar47/Desktop/thinking_parrot/tp2a-2 models/tp2_v0.0.1-2.hdf5')
output_char_num = 400
generate_text(EX_PATH, MASTER_PATH, model, output_char_num)
