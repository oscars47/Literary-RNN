# file to convert separate textfiles
# @oscars47

# imports
import os
from tqdm import tqdm
from dataprepsn import TextData
import numpy as np

# set main directory
MAIN_DIR = '/home/oscar47/Desktop/thinking_parrot'
TEXT_DIR = os.path.join(MAIN_DIR, 'texts')

# function to get excerpt of long texts if need be
def get_excerpt(filepath, num):
    if filepath.endswith('.txt'):
        with open(os.path.join(TEXT_DIR, filepath), 'r') as file:
            text = file.read()
            # grab first num characters
            text = text[:num]
        # now write as new file
        name = filepath.split('.txt')[0]
        with open(os.path.join(TEXT_DIR, name+'_short.txt'), 'w') as file:
            file.write(text)
            file.close()

# function to concatenate all textfiles together in the text dir
def concat_texts(dir):
    if not(os.path.exists(os.path.join(TEXT_DIR, 'master.txt'))): # create master file if there isnt one already
        master_string = '' # define master string which will will add all the text to
        for path in tqdm((os.listdir(dir))):
            if path.endswith('.txt'): # if it's a textfile
                # open file containing text
                with open(os.path.join(TEXT_DIR, path), 'r') as file:
                    text = file.read() # string object
                    master_string += text
                    file.close()
        # now write the string to a file
        with open(os.path.join(TEXT_DIR, 'master.txt'), 'w') as file:
            file.write(master_string)
            file.close()
                
concat_texts(TEXT_DIR) # run the function

# now we call dataprep2 to make a TextData object and prepare the text
# set maxChar limit; we will have a sandwhich pf this number of characters on either side
MASTER_TEXT_PATH = os.path.join(TEXT_DIR, 'master.txt')
# initialize text object
maxChar = 100
input=1 # shakespeare cleaning
master=TextData(MASTER_TEXT_PATH, input, maxChar)
x_train, y_train, x_val, y_val = master.prepare_data()

print(x_train)

# now save them
np.save(os.path.join(MAIN_DIR, 'texts_0.0.2_sn_prep', 'x_train.npy'), x_train)
np.save(os.path.join(MAIN_DIR, 'texts_0.0.2_sn_prep', 'y_train.npy'), y_train)
np.save(os.path.join(MAIN_DIR, 'texts_0.0.2_sn_prep', 'x_val.npy'), x_val)
np.save(os.path.join(MAIN_DIR, 'texts_0.0.2_sn_prep', 'y_val.npy'), y_val)

# TOAST_TEXT_PATH = os.path.join(TEXT_DIR, 'toaster_man.txt')
# # initialize text object
# maxChar = 100
# master=TextData(TOAST_TEXT_PATH, maxChar)
# x_train, y_train, x_val, y_val = master.prepare_data()

# # now save them
# np.save(os.path.join(MAIN_DIR, 'texts_prep', 'test', 'x_train.npy'), x_train)
# np.save(os.path.join(MAIN_DIR, 'texts_prep', 'test', 'y_train.npy'), y_train)
# np.save(os.path.join(MAIN_DIR, 'texts_prep', 'test', 'x_val.npy'), x_val)
# np.save(os.path.join(MAIN_DIR, 'texts_prep', 'test', 'y_val.npy'), y_val)