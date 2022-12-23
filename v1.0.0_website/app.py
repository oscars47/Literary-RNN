import gradio as gr
import numpy as np
import keras

# helper files----
# 1. clean data
# open the textfile; convert all text to lower case for ease of use
# takes in tetxfile path
def clean_data(text):
    # lowercase!!
    text = text.lower()

    # print('number of characters in textfile, including newline:', len(text))
    # remove all new line '\n' characters as these don't have any meaning
    # break up into list of characters; if the char is '\n' don't add it
    # then recompile into string

    # list of all the bad characters
    forbidden_char = ['\n', '\\', '^', '{', '|', '}', '~', '£', 
    '¥', '§', '©', '«', '¬', '®', '°', '»', '„', '•', '™', '■', '□', '►']

    temp = []
    i = 0
    while i < (len(text)-3):
        char = text[i]
        char_next = text[i+1]
        char_next_next = text[i+2]
        char_next_next_next = text[i+3]
        
        # if the next character isn't a new line and char isn't '\n', add it
        if not(char in forbidden_char):
            # check if next character is '¬'
            if char_next == '¬':
                if (char_next_next == '\n') or (char_next_next_next=='\n'):
                    temp.append(char)
                    i+=3
            
            elif not(char_next in forbidden_char):
                temp.append(char)
    
            # next char is newline
            elif char_next == '\n':
                if char != ' ':
                    temp.append(char)
                    temp.append(' ')
                else:
                    temp.append(char)
            else:
                temp.append(char)
        i+=1

        # make sure we don't forget to append final character if not illegal!!
        # print(i == len(text)-3)
        if (i == len(text)-3) and not(char_next in forbidden_char):
            temp.append(char_next)
            if not(char_next_next) in forbidden_char:
                temp.append(char_next_next)
                if not(char_next_next_next) in forbidden_char:
                    temp.append(char_next_next_next)

    #reset nasrudin string
    text = ''
    for char in temp:
        text += char

    # print('number of characters in textfile:', len(text))

    # return cleaned data file
    return text


# get nasrudin text cleaned
with open('sufis_full.txt', 'r') as file:
    text = file.read()
    nasrudin = clean_data(text)

# 2. helper function to parse string into alphabet and get mapping dictionaries from char to int and int to char
def parse_text(text):
    # first find all the unique characters; sort them
    alphabet = sorted(list(set(text)))

    # create a dictionary for a 1-1 map from character to integer and vice versa so we can seamlessly convert
    char_to_int = dict((c, i) for i, c in enumerate (alphabet))
    int_to_char = dict((i, c) for i, c in enumerate (alphabet))

    return alphabet, char_to_int, int_to_char

alphabet, char_to_int, int_to_char = parse_text(nasrudin)

# set max_Char value; this is length of sentence which we train on -- do not change this
global maxChar
maxChar=40

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

# helper function that we call to generate text
# takes in an input string, hdf5 trained model, and desired output length of text
model_types = ['Nasrudin', 'Shakespeare', 'Hemingway']

# function takes in input string, what text TP was trained on, and the text length as provided by huggingface input
def generate_text(input, text_len):
    # make sure at least 40 characters for training
    if len(input) < maxChar:
        raise gr.Error('Input must have >= %i characters. You have %i.' %(maxChar, len(input)))

    # make sure output num characters is integer
    if type(text_len) != int:
        raise gr.Error('Number of generated characters must be an integer!')

    # clean input data
    input = clean_data(input)

    # load desired model and set maxChar limit -- change these as we generate new models!
    
    model = keras.models.load_model('nasrudin_v1.0.0.hdf5')

    # grab last maxChar characters
    sentence = input[-maxChar:]

    # initalize generated string
    generated = ''
    generated += input
        
    # randomly pick diversity parameter
    diversities = [0.2, 0.5, 1.0, 1.2]
    div_index = int(np.random.random()*(len(diversities)))
    diversity = diversities[div_index]
    # print('diversity:', diversity)
    # sys.stdout.write(input)

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
    print()

    return generated

# call hugging space interactive interface; use Blocks

with gr.Blocks() as think:
    # have intro blurb
    gr.Markdown("Hi! I'm Thinking Parrot, a text generating AI! 🦜" )
    
    # have accordian blurb
    with gr.Accordion("Click for more details!"):
        gr.Markdown("Simply type at least 40 characters into the box labeled 'Your Input Text' below, choose what training text you like me to respond based on (e.g., Shakespeare -- n.b. this feature is currently in development and all options will use the Nasrudin training), and then select the number of output characters you want (note: try lower values for a faster response). Then click 'Think'! My response will appear in the box labeled 'My Response'.")
    
    # setup user interface
    input = [gr.Textbox(label = 'Your Input Text'), gr.Slider(minimum=40, maximum =200, label='Number of output characters', step=10)]
    output = gr.Textbox(label = 'My Response')
    think_btn = gr.Button('Think!')
    think_btn.click(fn= generate_text, inputs = input, outputs = output)

# enable queing if heavy traffic
think.queue(concurrency_count=3)
think.launch()