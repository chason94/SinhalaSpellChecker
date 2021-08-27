""" helper functions for
    - data loading
    - representation building
    - vocabulary loading
"""

from collections import defaultdict
import numpy as np
# import pickle
import random
# from random import shuffle
from tqdm import tqdm

############################################################

# #TODO: think of an open vocabulary system
# WORD_LIMIT = 9999 # remaining 1 for <PAD> (this is inclusive of UNK)
# task_name = ""
# TARGET_PAD_IDX = -1
# INPUT_PAD_IDX = 0

keyboard_mappings = None

############################################################

def get_lines(filename):
    print(filename)
    f = open(filename)
    lines = f.readlines()
    if "|||" in lines[0]:
        # remove the tag
        clean_lines = [line.split("|||")[1].strip().lower() for line in lines]
    else:
        clean_lines = [line.strip().lower() for line in lines]
    return clean_lines



def _get_line_representation(line, rep_list=['swap','drop','add','key'], probs=[0.25,0.25,0.25,0.25]):
    modified_words = []
    for word in line.split():
        rep_type = np.random.choice(rep_list, 1, p=probs)[0]
        if 'swap' in rep_type:
            # word_rep, new_word = get_swap_word_representation(word)
            new_word = get_swap_word_representation(word)
        elif 'drop' in rep_type:
            # word_rep, new_word = get_drop_word_representation(word, 1.0)
            new_word = get_drop_word_representation(word, 1.0)
        elif 'add' in rep_type:
            # word_rep, new_word = get_add_word_representation(word)
            new_word = get_add_word_representation(word)
        elif 'key' in rep_type:
            # word_rep, new_word = get_keyboard_word_representation(word)
            new_word = get_keyboard_word_representation(word)
        elif 'none' in rep_type or 'normal' in rep_type:
            # word_rep, _ = get_swap_word_representation(word)
            # new_word = word
            new_word = word
        else:
            #TODO: give a more ceremonious error...
            raise NotImplementedError
        # rep.append(word_rep)
        modified_words.append(new_word)
    # return rep, " ".join(modified_words)
    return " ".join(modified_words)

def get_line_representation(lines, rep_list=['swap','drop','add','key'], probs=[0.25,0.25,0.25,0.25]):
    # rep = []
    modified_lines = [_get_line_representation(line,rep_list,probs) for line in lines]
    return modified_lines


""" 
word representation from individual chars
    one hot (first char) + bag of chars (middle chars) + one hot (last char)
"""
def get_swap_word_representation(word):

    # dirty case
    if len(word) == 1 or len(word) == 2:
        # rep = one_hot(word[0]) + zero_vector() + one_hot(word[-1])
        # return rep, word
        return word

    # rep = one_hot(word[0]) + bag_of_chars(word[1:-1]) + one_hot(word[-1])
    if len(word) > 3:
        idx = random.randint(1, len(word)-3)
        word = word[:idx] + word[idx + 1] + word[idx] + word[idx+2:]

    # return rep, word
    return word

""" 
word representation from individual chars (except that one of the internal
    chars might be dropped with a probability prob
"""
def get_drop_word_representation(word, prob=0.5):
    p = random.random()
    if len(word) >= 5 and p < prob:
        idx = random.randint(1, len(word)-2)
        word = word[:idx] + word[idx+1:]
        # rep, _ = get_swap_word_representation(word) # don't care about the returned word
        _ = get_swap_word_representation(word) # don't care about the returned word
    elif p > prob:
        # rep, word = get_swap_word_representation(word)
        word = get_swap_word_representation(word)
    else:
        # rep, _ = get_swap_word_representation(word) # don't care about the returned word
        _ = get_swap_word_representation(word) # don't care about the returned word
    # return rep, word
    return word

def get_add_word_representation(word):
    if len(word) >= 3:
        idx = random.randint(1, len(word)-1)
        random_char = _get_random_char()
        word = word[:idx] + random_char + word[idx:]
        # rep, _ = get_swap_word_representation(word) # don't care about the returned word
        _ = get_swap_word_representation(word) # don't care about the returned word
    else:
        # rep, _ = get_swap_word_representation(word) # don't care about the returned word
        _ = get_swap_word_representation(word) # don't care about the returned word
    # return rep, word
    return word

def get_keyboard_word_representation(word):
    if len(word) >=3:
        idx = random.randint(1, len(word)-2)
        keyboard_neighbor = _get_keyboard_neighbor(word[idx])
        word = word[:idx] + keyboard_neighbor + word[idx+1:]
        # rep, _ = get_swap_word_representation(word) # don't care about the returned word
        _ = get_swap_word_representation(word) # don't care about the returned word
    else:
        # rep, _ = get_swap_word_representation(word) # don't care about the returned word
        _ = get_swap_word_representation(word) # don't care about the returned word
    # return rep, word
    return word

#TODO: is that all the characters we need??
def _get_random_char():
    alphabets = "abcdefghijklmnopqrstuvwxyz"
    alphabets = [i for i in alphabets]
    return np.random.choice(alphabets, 1)[0]


def _get_keyboard_neighbor(ch):
    global keyboard_mappings
    if keyboard_mappings is None or len(keyboard_mappings) != 26:
        keyboard_mappings = defaultdict(lambda: [])
        keyboard = ["qwertyuiop", "asdfghjkl*", "zxcvbnm***"]
        row = len(keyboard)
        col = len(keyboard[0])

        dx = [-1, 1, 0, 0]
        dy = [0, 0, -1, 1]

        for i in range(row):
            for j in range(col):
                for k in range(4):
                    x_, y_ = i + dx[k], j + dy[k]
                    if (x_ >= 0 and x_ < row) and (y_ >= 0 and y_ < col):
                        if keyboard[x_][y_] == '*': continue
                        if keyboard[i][j] == '*': continue
                        keyboard_mappings[keyboard[i][j]].append(keyboard[x_][y_])

    if ch not in keyboard_mappings: return ch
    return np.random.choice(keyboard_mappings[ch], 1)[0]
