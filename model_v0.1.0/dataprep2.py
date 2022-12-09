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
        # define list of forbidden characters
        forbidden_char = ['…', '\n', '\\', '^', '{', '|', '}', '~', '£', 
                            '¥', '§', '©', '«', '¬', '®', '°', '»', '„', 
                            '•', '™', '■', '□', '►', '\ufeff', '€', '>', '<', '=']

        # define new string
        clean_text = ''
        for char in text:
            if not(char in forbidden_char):
                clean_text += char

        # return cleaned textfile
        return clean_text
    
    # define getters
    def get_text(self):
        return self.clean_text
    
    #returns tuple of alphabet, char_to_int, int_to_char
    def get_parsed(self):
        return (self.alphabet, self.char_to_int, self.int_to_char)

    def get_maxChar(self):
        return self.maxChar
    
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
    
    # # helper function to compute the Gaussian
    # def gaussian(self, x, mean, stdev):
    #     return (1/(stdev * np.sqrt(2 * np.pi)) * np.exp( - (x - mean)**2 / (2 * stdev**2)))

    # helper function to plot normal dist: rtakes in mean, stdev, and list holding lens
    def make_gauss_scatter(self, mean, stdev, len_ls):
        plt.figure(figsize=(10,7))
        counts, edges, bars = plt.hist(len_ls, 30, color='magenta', alpha=0.8)
        plt.bar_label(bars)# add count labels
        # now plot the Gaussian
        # x_ls = np.linspace(0, self.maxChar-1, 1000)
        # y_ls = []
        # for x in x_ls:
        #     y_ls.append(self.gaussian(x, mean, stdev))
        # plt.plot(x_ls, y_ls, color='magenta', label='gaussian')
        plt.xlabel('Length of toast samples', fontsize=16)
        plt.ylabel('Number of samples', fontsize=16)
        plt.title('Toast lengths for toaster manual', fontsize=18)
        plt.show()

    # helper function to return the toast len
    def get_toast_len(self, mean, stdev):
        toast_len = int(np.random.normal(mean, stdev))
        return toast_len
    
    # helper function to generate sentences and target_chars for test and validation data
    def format_data(self, text):
        #print(type(self.text))
        # first determine lengths of semireduendant sentences; store these in a list
        sentences = []
        target_chars=[]
        len_ls = [] # define list to hold lengths used so we can plot
        # define stndard dev and mean for normal distr
        # stdev = (2/9)*(self.maxChar - 1)
        # mean = (2/3)*(self.maxChar - 1)
        stdev = (1/2)*(self.maxChar - 1)
        mean = (self.maxChar - 1)
        for i in range(1, len(text)-2): # 3 is the min number of characters to add; but we must add 21 bc range object goes to len-1
            # if i is the last possible index, then len must be 3
            # t0: toast 0, t1: toast 1, h: honey (target)
            if i == 1: # starting at index 1
                toast_len=1
                t0 = text[0]# half 0
                t1 = text[2] # second half
                h = text[1] # target sits in middle
                sentences.append(t0+t1)
                target_chars.append(h)
            elif i == len(self.text)-3:
                toast_len=1
                t0 = text[-3]# half 0
                t1 = text[-1] # second half
                h = text[-2] # target sits in middle
                sentences.append(t0+t1)
                target_chars.append(h)
            else:
                # compute len, following normal distribution between 1 and maxChar; will go from [:num] as first part then [num+1:] concatenated; predict at num
                # need toastlen positive but no more than  maxChar
                goodtoast = False
                while goodtoast==False:
                    toast_len = self.get_toast_len(mean, stdev)
                    # add 1 to len since the distr here goes from 0 up to maxChar-1
                    toast_len+=1
                    #print(toast_len)
                    if (toast_len > 0) and (toast_len <= self.maxChar): # if get acceptable toast, can leave
                        goodtoast=False
                        break

                # now need to check that we dont run over
                if (toast_len > (len(text) - i)) or (toast_len > i): # need to check we dont over or try to go back father than we've come
                    tryagain=True
                    while tryagain==True: # need to keep subtracting by 1 and seeing if this works
                        toast_len-=1
                        if (toast_len <= (len(text) - i)) or (toast_len <= i):
                            tryagain=False
                            break
                # now we can append len 
                t0 = text[i-toast_len:i]
                t1 = text[i+1:i+toast_len]
                h = text[i]
                sentences.append(t0+t1)
                target_chars.append(h)
            len_ls.append(toast_len)

        # make scatter plot of values to verify, with superimposed plot
        self.make_gauss_scatter(mean, stdev, len_ls)
        
        # now need to create padded binary datasets: move array into middle of bigger array; then loop through indices; if it's a valid character will get converted
        # can append a forbidden character on either ends to fill this sandwhich, that way it gets mapped to a column of 0s since we first iniitalize a 0 array and then would go back to make it 1-hot
        complete_sentences = []
        for i, sentence in tqdm(enumerate(sentences), desc='progress loading sentences...', position=0):
            # 1. compute difference from maxChar and len/2
            diff = self.maxChar - int(len(sentence)/2)
            # 2. initialize new string for each sentence
            complete_sentence = ''
            for i in range(diff):
                complete_sentence+='£' # appending forbidden
            # 3. now add 'real' sentence
            complete_sentence+=sentence
            # 4. append forbidden again
            for i in range(diff):
                complete_sentence+='£'
            complete_sentences.append(complete_sentence)

        # initialize all 0s
        data = np.zeros((len(complete_sentences), self.maxChar*2, len(self.alphabet)), dtype = np.uint8)
        labels = np.zeros((len(complete_sentences), len(self.alphabet)), dtype = np.uint8)
        
        # now go back through the complete sentences and fill in 1-hots
        for i, sentence in enumerate(complete_sentences):
            for j, char in enumerate(sentence):
                if char != '£': # encode 1 iff it's not padded
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