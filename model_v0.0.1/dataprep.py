# file to handle data prep for Thinking Parrot
# @oscars47

import os
import numpy as np
import pandas as pd

class TextData:
    # define constructor: pass path to text, index in list of possible clean data functions: [nasrudin, shakespeare]
    def __init__(self, path, index, maxChar):
        # open file containing text
        with open(path, 'r') as file:
            text = file.read()
            # clean the text according to index assignment
            if index == 1:
                self.clean_text = clean_data_shakespeare(text)
            else:
                self.clean_text = clean_data_nasrudin(text)
        # set environ variable for the text
        os.environ['TEXT'] = self.clean_text
        # get alphabet and dictionaries
        self.alphabet, self.char_to_int, self.int_to_char = self.parse_text(self.clean_text)
        #set maxChar
        self.maxChar = maxChar

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
    def parse_text(self, text):
        # first find all the unique characters; sort them
        alphabet = sorted(list(set(text)))
        #print('characters in alphabet:', len(text))

        # create a dictionary for a 1-1 map from character to integer and vice versa so we can seamlessly convert
        char_to_int = dict((c, i) for i, c in enumerate (alphabet))
        int_to_char = dict((i, c) for i, c in enumerate (alphabet))
        #print('char to int dictionary:',char_to_int)

        return alphabet, char_to_int, int_to_char

    # helper function to generate sentences and target_chars for test and validation data
    def format_data(self, text):
        # generate list of sentences and the target character following the sentence
        sentences = []
        target_chars = []

        # for loop to populate lists of semiredudant sentences
        for i in range(len(text) - self.maxChar):
            sentences.append(text[i: i + self.maxChar])
            target_chars.append(text[i + self.maxChar])
        #print('number of semiredudant sentences:', len(text))
        #print('example sentence:', sentences[0])

        # convert to binary arrays
        data = np.zeros((len(sentences), self.maxChar, len(self.alphabet)), dtype = np.bool_)
        labels = np.zeros((len(sentences), len(self.alphabet)), dtype = np.bool_)

        # set a 1 in the index corresponding to the integer mapping of the character
        for i, sentence in enumerate(sentences):
            for j, char in enumerate(sentence):
                data[i, j, self.char_to_int[char]] = 1
            labels[i, self.char_to_int[target_chars[i]]] = 1

        return data, labels

    # prepare data wil call format data; master function for training preparating
    def prepare_data(self):
        text = self.clean_text
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

        return x_train, y_train, x_val, y_val
    # find all words in string
    def do_analysis(self):
        text = self.clean_text
        # get words
        words = text.split(' ')
        # get unique words
        words_unique = list(set(words))
        print('there are %i unique words' %(len(words_unique)))

        # find most used words
        # create pandas df with word and number of counts
        word_count_df = pd.DataFrame({'word': [], 'count':[]})
        for word in words_unique:
            # don't want to count space
            if word != ' ':
                count = words.count(word)
                word_count_df = word_count_df.append({'word': word, 'count': count}, ignore_index=True)
        
        word_count_df = word_count_df.sort_values(by = 'count', ascending=False) 
        #print(word_count_df)

        #find unique phrases (distinct combinations of three sequential words)
        phrase_count_df = pd.DataFrame({'phrase': [], 'count': []})
        for i in range(0, len(words)-3, 3):
            
            if  (((words[i] != ' ') and (words[i+1] != ' ') and (words[i+2] != ' ')) and ((words[i] != '') and (words[i+1] != '') and (words[i+2] != ''))):        
                phrase = words[i] + ' ' + words[i+1] + ' ' + words[i+2]

                # don't want phrase of multiple spaces
                
                # if phrase doesn't show up, add it
                if len(phrase_count_df.loc[phrase_count_df['phrase']==phrase]) == 0:
                    phrase_count_df = phrase_count_df.append({'phrase': phrase, 'count': 1}, ignore_index=True)
                # we have repeat, so add it
                else:
                    # find index
                    index = phrase_count_df.loc[phrase_count_df['phrase']==phrase].index.values[0]
                    # now increment count
                    count = phrase_count_df.iat[index, 1]
                    count+=1
                    phrase_count_df.iat[index, 1] = count

        phrase_count_df = phrase_count_df.sort_values(by = 'count', ascending=False)

        print('the most used word is "%s", used %i times'%(word_count_df['word'].values[0], word_count_df['count'].values[1]))
        print('the most used phrase is "%s", used %i times'%(phrase_count_df['phrase'].values[0], phrase_count_df['count'].values[0]))
        
        return words_unique, word_count_df, phrase_count_df

class TextDataText:
    # define constructor: pass path to text, index in list of possible clean data functions: [nasrudin, shakespeare]
    def __init__(self, text, index, maxChar):
    
        if index == 1:
            self.clean_text = clean_data_shakespeare(text)
        else:
            self.clean_text = clean_data_nasrudin(text)
    
        # get alphabet and dictionaries
        self.alphabet, self.char_to_int, self.int_to_char = self.parse_text(self.clean_text)
        #set maxChar
        self.maxChar = maxChar

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
    def parse_text(self, text):
        # first find all the unique characters; sort them
        alphabet = sorted(list(set(text)))
        #print('characters in alphabet:', len(text))

        # create a dictionary for a 1-1 map from character to integer and vice versa so we can seamlessly convert
        char_to_int = dict((c, i) for i, c in enumerate (alphabet))
        int_to_char = dict((i, c) for i, c in enumerate (alphabet))
        #print('char to int dictionary:',char_to_int)

        return alphabet, char_to_int, int_to_char

    # helper function to generate sentences and target_chars for test and validation data
    def format_data(self, text):
        # generate list of sentences and the target character following the sentence
        sentences = []
        target_chars = []

        # for loop to populate lists of semiredudant sentences
        for i in range(len(text) - self.maxChar):
            sentences.append(text[i: i + self.maxChar])
            target_chars.append(text[i + self.maxChar])
        #print('number of semiredudant sentences:', len(text))
        #print('example sentence:', sentences[0])

        # convert to binary arrays
        data = np.zeros((len(sentences), self.maxChar, len(self.alphabet)), dtype = np.bool_)
        labels = np.zeros((len(sentences), len(self.alphabet)), dtype = np.bool_)

        # set a 1 in the index corresponding to the integer mapping of the character
        for i, sentence in enumerate(sentences):
            for j, char in enumerate(sentence):
                data[i, j, self.char_to_int[char]] = 1
            labels[i, self.char_to_int[target_chars[i]]] = 1

        return data, labels

    # prepare data wil call format data; master function for training preparating
    def prepare_data(self):
        text = self.clean_text
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

        return x_train, y_train, x_val, y_val

# functions to clean input data
def clean_data_nasrudin(text):
    #lowercase!
    text = text.lower()
    # print('number of characters in textfile, including newline:', len(text))
    # remove all new line '\n' characters as these don't have any meaning
    # break up into list of characters; if the char is '\n' don't add it
    # then recompile into string

    # list of all the bad characters
    forbidden_char = ['…', '\n', '\\', '^', '{', '|', '}', '~', '£', 
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

    #reset string
    text = ''
    for char in temp:
        text += char

# print('number of characters in textfile:', len(text))

# return cleaned data file
    return text

# yes '\n'
def clean_data_shakespeare(text):
    text = text.lower()
    # print('number of characters in textfile, including newline:', len(text))
    # remove all new line '\n' characters as these don't have any meaning
    # break up into list of characters; if the char is '\n' don't add it
    # then recompile into string

    # list of all the bad characters
    forbidden_char = ['…', '\t', '\\', '^', '{', '|', '}', '~', '£', 
    '¥', '§', '©', '«', '¬', '®', '°', '»', '„', '•', '™', '■', '□', '►']

    temp = []
    i = 0
    for char in text:
        if not(char in forbidden_char):
            temp.append(char)

    #reset main string
    text = ''
    for char in temp:
        text += char

# print('number of characters in textfile:', len(text))

    # return cleaned data file
    return text