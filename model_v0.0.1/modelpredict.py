# file to handle the prediction based on a trained model
# @oscars47

from dataprep import *

import keras


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

def generate_text(input, text_len, td):
    initialize_pre(td)
    print(alphabet)
    # load in model
    model = keras.models.load_model('nasrudin_v1.0.0.hdf5')
    '''
    # make sure at least 40 characters for training
    if len(input) < maxChar:
        raise ValueError('Input must have >= %i characters. You have %i.' %(maxChar, len(input)))
    '''
    # grab last maxChar characters
    #sentence = input[-maxChar:]
    sentence = input
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
        # sys.stdout.write(next_char)
        # sys.stdout.flush()
    #print()
    '''
    with open("experimental_log.txt", "a") as file:
        write_stuff = ['RNN: cc = %i, diversity = %i' %(text_len, diversity), '\n', generated, '\n', '\n']
        file.writelines(write_stuff)
    '''
    return generated

# function to call generate model >1 times
def experiment_RNN(input, model, text_len, num, maxChar, alphabet, char_to_int, int_to_char):
    for i in range(0, num):
        generate_text(input, model, text_len, maxChar, alphabet, char_to_int, int_to_char)
        print()

    # can also potentially customsize text_len parameter