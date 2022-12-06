# file to convert separate textfiles
# @oscars47

# imports
import os
from tqdm import tqdm
from dataprep2 import TextData

# set main directory
MAIN_DIR = '/home/oscar47/Desktop/thinking_parrot'
TEXT_DIR = os.path.join(MAIN_DIR, 'texts')

# function to concatenate all textfiles together in the text dir
def concat_texts(dir):
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
            
#concat_texts(TEXT_DIR) # run the function

# now we call dataprep2 to make a TextData object and prepare the text
# set maxChar limit; we will have a sandwhich pf this number of characters on either side
MASTER_TEXT_PATH = os.path.join(TEXT_DIR, 'toaster_man.txt')
# initialize text object
maxChar = 500
master=TextData(MASTER_TEXT_PATH, maxChar)
master.format_data()