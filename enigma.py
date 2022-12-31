# file to do encryption and reordering
import numpy as np

ALPHABET = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 
'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 
'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 
'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 
'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '.', ',', ';', ':', '—', '!', '#', 
'$', '%', '&', '(', ')', '*', '/', '?', '@', '[', 
']', '^', '-', '_', '{', '|', '}', "'"]

# encrypt-------------------------
def encrypt(input, key):
    output='' # initialize output
    for i, char in enumerate(input):
        if char != '\n' and char != '\t' and char != ' ':
            shift = key[i] # get key
            output+= ALPHABET[(ALPHABET.index(char)+shift)%len(ALPHABET)]
            # print(output)
        else:
            output+=char
        
    return output

def decrypt(input, key):
    output='' # initialize output
    for i, char in enumerate(input):
        if char != '\n' and char != '\t' and char != ' ':
            shift = key[i] # get key
            output+= ALPHABET[(ALPHABET.index(char)-shift)%len(ALPHABET)]
            # print(output)
        else:
            output+=char
    return output

# for reordering----------------------------
# Simultaneous must be used only when each element in key is unique
def buildDicts(input, key):
    encr_dict = dict(zip(list(range(len(input))), key))
    decr_dict = dict(zip(key, list(range(len(input)))))
    # print(encr_dict)
    # print(decr_dict)
    return encr_dict, decr_dict

# function to hold pairs of switches
def buildPairs(input, key):
    encr_ls = list(zip(list(range(len(input))), key))
    decr_ls = list(zip(key, list(range(len(input)))))
    # reverse decr list
    decr_ls_new = [decr_ls[i] for i in range(len(decr_ls)-1, -1, -1)]
    #print('dec', decr_ls_new)
    
    return encr_ls, decr_ls_new

def reorderSimultaneous(input, key):
    output = [' ' for i in range(len(input))] # initialize output vector
    encr_dict, _ = buildDicts(input, key) # get the dictionary
    for i, c in enumerate(input):
        output[encr_dict[i]] = c # move characters where they need to go according to key
    output_str = ''
    for c in output:
        output_str+=c
    return output_str

def rereorderSimultaneous(input, key):
    output = [' ' for i in range(len(input))] # initialize output vector
    _, decr_dict = buildDicts(input, key) # get the dictionary
    for i, c in enumerate(input):
        output[decr_dict[i]] = c # move characters where they need to go according to key
    output_str = ''
    for c in output:
        output_str+=c
    return output_str

def reorderSequential(input, key):
    input_ls = [c for c in input]
    encr_ls, _ = buildPairs(input, key) # get the dictionary
    for i, c in enumerate(input_ls):
        index = encr_ls[i][1]
        #print(i, index, c)
        input_ls.insert(index, input_ls.pop(i))
        #print(input_ls)
    output=''
    for c in input_ls:
        output+=c
    return output

def rereorderSequential(input, key):
    input_ls = [c for c in input]
    _, decr_ls = buildPairs(input, key) # get the dictionary
    for i in range(len(input_ls)): 
        current_index = decr_ls[i][0]
        next_index = decr_ls[i][1]
        c = input_ls[current_index]
        #print(current_index, next_index, c)
        input_ls.insert(next_index, input_ls.pop(current_index))
        #print(input_ls)
    output=''
    for c in input_ls:
        output+=c
    return output


# define your input here! you can define an English input and encode it by reordering and/or encrypting using the functions above
# or you can enter an encoded input and call from the functions above to decode the message. you can follow the example below!
input = "hello world!"
# key = [(i+np.random.randint(0, len(input)))%len(input) for i in range(len(input))]
# key2 = [np.random.randint(0, len(input)) for i in range(len(input))]
key = [(i+i)%len(input) for i in range(len(input))] #key holds reordering key
key2 = [i for i in range(len(input))] # key2 is for encrypting

input2 = encrypt(input, key2)

input3 = reorderSequential(input2, key)
print('encoded:', input3)

# input3 represents the complete encrypted and reordered message. now we can undo this encoding.

input4 = rereorderSequential(input3, key)
input5 = decrypt(input4, key2)
print('original:', input5)
