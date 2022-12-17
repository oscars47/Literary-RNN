# file to hold TextData object to handle data processing for TP2
# @ oscars47

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# create the class
class TextData:
    # define the constructor--------------------
    def __init__(self, path, maxChar):
        with open(path, 'r') as file:
            text = file.read()
            # now clean the text
            self.text = self.clean_data(text)
        # initialize maxChar attribute
        self.maxChar = maxChar
        # initialize alphabet and dictionaries
        self.alphabet, self.char_to_int, self.int_to_char = self.parse_text()
        print(self.alphabet)
    # function to clean the data-----------------
    def clean_data(self, text):
        # lowercase
        text = text.lower()
        # define list of forbidden characters
        forbidden_char = ['…', '\\', '^', '{', '|', '}', '~', '£', 
                            '¥', '§', '©', '«', '¬', '®', '°', '»', '„', 
                            '•', '™', '■', '□', '►', '\ufeff', '€', '>', '<', '=']

        # define new string
        clean_text = ''
        for char in text:
            if not(char in forbidden_char):
                clean_text += char

        # return cleaned textfile
        return clean_text
    
    # open the textfile; convert all text to lower case for ease of use
    # takes in tetxfile path

    # open the textfile; convert all text to lower case for ease of use
    # takes in tetxfile path

    # helper function to parse string into alphabet and get mapping dictionaries from char to int and int to char
    def parse_text(self):
        # first find all the unique characters; sort them
        alphabet = sorted(list(set(self.text)))
        #print('characters in alphabet:', len(text))

        # create a dictionary for a 1-1 map from character to integer and vice versa so we can seamlessly convert
        char_to_int = dict((c, i) for i, c in enumerate (alphabet))
        int_to_char = dict((i, c) for i, c in enumerate (alphabet))
        #print('char to int dictionary:',char_to_int)

        return alphabet, char_to_int, int_to_char
    
    # helper function to generate sentences and target_chars for test and validation data
    def format_data(self, text):
        #print(type(self.text))
        # first determine lengths of semireduendant sentences; store these in a list
        sentences = []
        target_chars=[]
    
        for i in range(self.maxChar, len(text)-self.maxChar): # maxChar is the min number of characters to add; but we must add 21 bc range object goes to len-1
        
            # now we can append len 
            t0 = text[i-self.maxChar:i]
            t1 = text[i+1:i+self.maxChar]
            h = text[i]
            sentences.append(t0+t1)
            target_chars.append(h)


        # initialize all 0s
        data = np.zeros((len(sentences), self.maxChar*2, len(self.alphabet)), dtype = np.uint8)
        labels = np.zeros((len(sentences), len(self.alphabet)), dtype = np.uint8)
        
        # now go back through the complete sentences and fill in 1-hots
        for i, sentence in enumerate(sentences):
            for j, char in enumerate(sentence):
                data[i, j, self.char_to_int[char]] = 1
            labels[i, self.char_to_int[target_chars[i]]] = 1  # since tagrget list is unaffected we're bing chilling

        return data, labels

    # now a final function to put it all together
    def prepare_data(self):
        text = self.text
        # split data into train and validation -- 80%, 20%
        index = int(len(text)*0.8)
        text_test = text[:index]
        text_val = text[index:]

        # x represents the input data, y the output targets
        #print('for training data:')
        x_train, y_train = self.format_data(text_test)
        #print('---------------')
        #print('for validation data:')
        x_val, y_val = self.format_data(text_val)

        # print for validation
        print('x_train', x_train.shape)
        print('y_train', y_train.shape)
        print('x_val', x_val.shape)
        print('y_val', y_val.shape)

        return x_train, y_train, x_val, y_val