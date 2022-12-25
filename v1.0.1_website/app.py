# file to run wesbite for 1.1.0 version
# using gradio for GUI
import gradio as gr
import numpy as np
import keras, sys

# call custom scripts
from dataprep import TextData, TextDataText


# read in mastertext DataPrep obj
index = 1 # for shakespeare cleaning
maxChar = 100 # based on model 
MASTER_PATH = 'master.txt'
# read in model
model = keras.models.load_model('model_1.0.1.hdf5')

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

def generate_text_text(input, text_len):
    index =1 #for shakes cleaning
    maxChar=100
    td = TextData(MASTER_PATH, index, maxChar)   
    alphabet = td.alphabet
    int_to_char = td.int_to_char
    char_to_int = td.char_to_int

    # input is the cleaned text
    td_sample = TextDataText(input, index, maxChar)
    input = td_sample.clean_text
    
    # make sure at least 40 characters for training
    if len(input) < 3:
        raise ValueError('Input must have >= 3 characters. You have %i.' %(maxChar, len(input)))
    print('input:')
    print(input)
    print('-----------------')
    print('output:')
     # need to prepare input
    if len(input) >= maxChar:
        # grab last maxChar characters
        sentence = input[-maxChar:]
    else:
        sentence = '' # initialize sentence
        # compute diff
        diff = maxChar - len(input)
        for i in range(diff):
            sentence+='£'
        sentence+=input
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
            if char != '£':
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

# call hugging space interactive interface; use Blocks

with gr.Blocks() as think:
    # have intro blurb
    gr.Markdown("Hi! I'm Thinking Parrot 1.0.1, a text generating AI! 🦜" )
    
    # have accordian blurb
    with gr.Accordion("Click for more details!"):
        gr.Markdown("Simply type at least 3 characters into the box labeled 'Your Input Text' below and then select the number of output characters you want (note: try lower values for a faster response). Then click 'Think'! My response will appear in the box labeled 'My Response'.")
    
    # setup user interface
    input = [gr.Textbox(label = 'Your Input Text'), gr.Slider(minimum=10, maximum =400, label='Number of output characters', step=10)]
    output = gr.Textbox(label = 'My Response')
    think_btn = gr.Button('Think!')
    think_btn.click(fn= generate_text_text, inputs = input, outputs = output)

# enable queing if heavy traffic
think.queue(concurrency_count=3)
think.launch()

#for testing
# input = input('enter text')
# generate_text_text(input, 400)