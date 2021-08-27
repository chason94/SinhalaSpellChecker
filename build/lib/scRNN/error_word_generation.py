# This python file implements errors classes common in sinhala language

import enum
import numpy as np
import re
import logging
import os
import random

from tqdm import tqdm
from datetime import datetime
from typing import Generator, Pattern, Union
from collections import Counter
import pickle
from sinling import SinhalaTokenizer
from .constant import Sinhala_alpha, symbols, vowel_symbols
from .helper import WeightedTuple

DEBUG = False
# vowel_symbols = {k : v for k, v in vowel_symbols}
replacement_dict = {
    # "න": "ණ", --
    "ණ": "න",

    # "ල": "ළ", --
    "ළ": "ල",

    "ධ": "ද",
    "ඛ": "ක",
    "ඝ": "ග",
    "ඡ": "ච",
    "ඨ": "ට",
    "භ": ["බ", "ඔ"],
    "ඵ": "ප",
    "ඩ": "ඪ",

    "ැ": "ෑ",
    "ෑ": "ැ",

    "ු": "ූ",
    "ූ": "ු",

    "ි": "ී",
    "ී": "ි",

    "ඟ": "ග",
    "ඦ": "ජ",
    "ඬ": "ඩ",
    "ඳ": "ද",
    "ඹ": "බ"
}

error_type_freq = {
    'diacritic':{
        'freq': 0.3,
        'pattern' : {
            
            # '(?![අ])[\u0D80-\u0DFF]': "ා",
            # '(?![අ])[\u0D80-\u0DFF]': "ෑ",
            # '(?![අ])[\u0D80-\u0DFF]': "ැ",

            "ැ": "ෑ",
            "ෑ": "ැ",

            "ු": "ූ",
            "ූ": "ු",

            "ි": "ී",
            "ී": "ි",

            "ෞ": "ො",
            "ො": "ෞ",

            "ෳ" : "ෟ",
            "ෟ" : "ෳ",

            "ෙ" : "ේ",
            "ේ" : "ෙ",

            "ො" : "ෝ",
            "ෝ" : "ො"
        } 
    },

    "na-Na-la" :{
        "freq" : 0.06,
        "pattern" :{
            "න": "ණ", 
            "ණ": "න",
            "ල": "ළ",
            "ළ": "ල",
        }
    },
    "prenasalized" : {
        "freq" : 0.06,
        "pattern" :{
            "ඟ": "ග",
            "ඦ": "ජ",
            "ඬ": "ඩ",
            "ඳ": "ද",
            "ඹ": "බ",
            "ග" : "ඟ" ,
            "ජ" : "ඦ" ,
            "ඩ" : "ඬ" ,
            "ද" : "ඳ" ,
            "බ" : "ඹ" 
        }
    },

    "similar_shape" : {
        "freq" : 0.06,
        "pattern" : {
            "ඞ" : "ඬ",
            "ඞ" : "ඩ",
            "ඩ" : "ඞ",
            "ඬ" : "ඞ",
            "ඡ" : "ජ",
            "ජ" : "ඡ",
            "සෘ" : "ඍ",
            "ඍ" : "සෘ",
        }		
    },
    "aspirated_and_unaspirated_consonant" : {
        "freq" : 0.04,
        "pattern" :{
            "ධ": "ද",
            "ඛ": "ක",
            "ඝ": "ග",
            "ඡ": "ච",
            "ඨ": "ට",
            "භ": "බ",
            "ඵ": "ප",
            "ඩ": "ඪ",
            "ථ": "ත",
            # "ජ": "ඣ",
            "ද": "ධ" ,
            "ක": "ඛ" ,
            # "ග": "ඝ" ,
            "ච": "ඡ" ,
            "ට": "ඨ" ,
            "බ": "භ" ,
            "ප": "ඵ" ,
            "ඪ": "ඩ" ,
            "ත": "ථ" ,
            "ඣ": "ජ" 
        }
    },
    "retroflexand_palatal_sibilant" : {
        "freq" : 0.02,
        "pattern" : {
            "ශ": "ෂ",
            "ෂ": "ශ",
            "ස": "ශ",
            "ස": "ෂ",
            "ශ": "ස",
            "ෂ": "ස",
        }        
    },
    'encoding' : {
        "freq" : 0.02,
        "pattern" : {
            # 'අා' to 'ආ'
            "ආ" : 'අා' ,
            # 'අැ', 'ඇ'
            'ඇ' :'අැ' ,
            "ඡ" : 'ඡු',
            # අෑ ඈ
            'ඈ' : 'අෑ',
            # ඒ ඒ
            'ඒ' : 'එ්',
            # 'ෙ', 'ච', 'ෙ'
            # '([\u0D99-\u0DC6])(\u0DDB)' : '\u0DD9\g<1>\u0DD9' ,
            # 'ැ', 'ු'
            'රැ' : 'රැු',
            # උෟ, ඌ
            'ඌ' : 'උෟ',
            vowel_symbols["ො"] : '\u0DD9\u0DCF' ,
            #  "ෝ" 
            vowel_symbols["ෝ"] :'\u0DD9\u0DCF\u0DCA' ,
            # "ු" : "ුුු",
            # "ූ": "ූූූූ",
            # vowel_symbols["ෝ"] :'\u0DD9\u0DCA\u0DCF' ,
            # vowel_symbols["ෝ"] :'\u0DDC\u0DCA' ,
        }  
    },
    
    'other' : {
        "freq" : 0.02,
        "pattern" : {
            "ඤ": "ඥ" ,
            "ඥ": "ඤ" ,
            'ඣ': 'ජ',
            'ං' :'න්',   
        }
    },
    'deletion': {
        'freq' : 0.02,
        'pattern' :{
           'ා':'',
           'ැ':'',
           'ෑ':'',
           'ි':'',
           'ී':'',
           'ු':'',
           'ූ':'',
           '්':''
        }
    },
    'encoding_0' : {
        "freq" : 0.02,
        "pattern" : {
            vowel_symbols["ෝ"] :'\u0DD9\u0DCA\u0DCF' ,
        }
    },
    'encoding_1' : {
        "freq" : 0.02,
        "pattern" : {
            vowel_symbols["ෝ"] :'\u0DDC\u0DCA' ,
        }
    }

}



data_dim = (len(Sinhala_alpha) + len(symbols)) * 3


class ErrorType(enum.Enum):
    Replace = enum.auto()
    Delete = enum.auto()
    Insert = enum.auto()
    D = enum.auto()
    E = enum.auto()
    F = enum.auto()
    G = enum.auto()
    H = enum.auto()
    I = enum.auto()

class JumbleType(enum.Enum):
    BEG = enum.auto()
    MID = enum.auto()
    END = enum.auto()





class AddNoise():
    def __init__(self, txt_path, **kwargs) -> None:
        with open(txt_path, mode='r', encoding='utf8') as f:
            text = f.read()
        remove_numbers = text 
        
        words = re.findall('[' + ''.join(Sinhala_alpha) + ''.join(vowel_symbols) + ']+', remove_numbers)
        self.words = words
        self.txt = txt_path
        self.errors = error_type_freq

        self.noise_frew = {
                            self.passThrough : 50,
                            self.natural : 41,
                            self.insertion_0 : 1, # adding consonats
                            self.insertion_1 : 2, # adding diacrtics
                            self.deletion : 3, 
                            self.jumble : 3, 
                            }
        # self.noise_frew = {
        #             self.passThrough : 80,
        #             self.natural : 0,
        #             self.insertion_0 : 5,
        #             self.insertion_1 : 5, 
        #             self.deletion : 5, 
        #             self.jumble : 5, 
        #             }
        self.create_erros_gen = WeightedTuple( self.noise_frew)



    @staticmethod
    def getAddNoiseObj(txt_path : str) -> 'AddNoise':
        
        return AddNoise(txt_path)

    def vocab(self):
        return set(self.words)
    def freq(self):
        c_dict = Counter(self.words)
        self.freq_dict =  {k : c_dict[k] for k in tqdm(self.vocab())}
        return self.freq_dict
    # filters
    def passThrough(self, word : str) -> str:
        return word

    def encodingErrors(self, word : str) -> str:
        return word
    def insertion_0(self, word : str) -> str:
        pos = random.randint(0, len(word))
        return word[:pos] + list(Sinhala_alpha)[np.random.randint(0,len(Sinhala_alpha))] + \
           word[pos:] if len(word) > 1 and pos < len(word) else word
    def insertion_1(self, word : str) -> str:
        pos = random.randint(0, len(word))
        return word[:pos] + list(vowel_symbols.values())[np.random.randint(0,len(vowel_symbols.values()))] + \
           word[pos:] if len(word) > 1 and pos < len(word) else word

    def deletion(self, word : str) -> str:
        pos = random.randint(0, len(word))
        return word[:pos] + word[pos + 1:] if len(word) > 1 and pos < len(word) else word

    def jumble(self, word : str) -> str:
        jumble_type = random.choice([JumbleType.BEG, JumbleType.MID, JumbleType.END])

        return word[0] + ''.join(random.sample(word[1:-1], len(word[1:-1]))) +  \
            word[-1] if jumble_type == JumbleType.MID else \
                (word[0] + ''.join(random.sample(word[1:], len(word[1:]))) if jumble_type == JumbleType.END else 
                ''.join(random.sample(word[:-1], len(word[:-1]))) + word[-1]
                ) if len(word) > 3 else word


    def __checkErrors(self, word: str) -> dict:
        return [(key,  [rex for rex in self.errors[key]['pattern'].keys() if re.search(rex, word)!= None if re.search(rex, word).groups() != None]) 
        for key in self.errors if any(re.search(rex, word).groups() != None for rex in self.errors[key]['pattern'].keys() if re.search(rex, word) != None)]


    def natural(self, word) -> str:
        error_type_lst =  self.__checkErrors(word)
        if len(error_type_lst) > 1:
            error_type = random.choice(error_type_lst)
            
            char = random.choice(error_type[1])
            if DEBUG:
                print(char, error_type)
            if len(char) > 1:
                
                word = re.subn(char, self.errors[error_type[0]]['pattern'][char], word, 1)[0]
            else:
                pos = random.choice([i for i, c in enumerate(word,0, ) if re.search(char, c) != None])

                word = word[:pos] + self.errors[error_type[0]]['pattern'][char] + \
                    word[pos + 1:] if 'අ' not in char else word[:pos] + \
                    self.errors[error_type[0]]['pattern'][char] + word[pos + 1:]
        return word


    def addNoiseFile(self, ofile_path) -> None:
        count = 0
        error_count = 0
        with open(self.txt, mode='r', encoding='utf8') as f:
            readlines = f.readlines()

        print('creating word dictionary')
        noise_filter = {k : [] for k in list(self.vocab())}
        
        with open(os.path.join(ofile_path, 'noisy.txt'), mode='w+', encoding='utf8') as f_noisy:
            with open(os.path.join(ofile_path, 'original.txt'), mode='w+', encoding='utf8') as f_orignal:
                with open(os.path.join(ofile_path, '_log.txt'), mode='w+', encoding='utf8') as f_log:
                    for line in tqdm(readlines): 
                        tokenized = re.sub('([^\u0D80-\u0DFF\u200a-\u200d]+)', ' \g<1> ', line.strip()).split()
                        for word in tokenized:
                            nword = word
                            count += 1
                            s_word = max(re.split(r'[^\u0D80-\u0DFF\u200a-\u200d]', word), key=len)
                            if len(s_word) > 1:
                                try :
                                    filter = random.choice(self.create_erros_gen) 
                                    nword = filter(nword)
                                    
                                except IndexError:
                                    print(s_word)
                                    raise IndexError
                            f_noisy.write("{} ".format(nword))
                            f_orignal.write("{} ".format(word))
                            if nword != word:
                                error_count += 1
                                f_log.write(' {} => {} ,'.format(word, nword))
                        
                        f_noisy.write("{} ".format('\n'))
                        f_orignal.write("{} ".format('\n'))
                    f_log.write('\nPercentage {}%'.format((error_count/count)*100))

class cyclic_iter():
    def __init__(self,lst, true_value):
        # print(lst)
        self.lst = lst
        self.lst.remove(true_value)
        self.true_value = true_value
        self.index = -1
        self.noise_frew = {
                    self.passthrough : 50,
                    # self.natural : 41,
                    # self.insertion_0 : 1,
                    # self.insertion_1 : 2, 
                    # self.deletion : 3, 
                    # self.jumble : 3, 
                    self.insertion_0 : 12,
                    self.insertion_1 : 12, 
                    self.deletion : 12, 
                    self.jumble : 12, 
                    }
        self.create_erros_gen = WeightedTuple( self.noise_frew)

    def passthrough(self) -> str:
        return self.true_value

    def natural(self):
        self.index += 1
        return self.lst[self.index % len(self.lst)]

    def insertion_0(self) -> str:
        word = self.true_value
        pos = random.randint(0, len(word))
        return word[:pos] + list(Sinhala_alpha)[np.random.randint(0,len(Sinhala_alpha))] + \
           word[pos:] if len(word) > 1 and pos < len(word) else word

    def insertion_1(self) -> str:
        word = self.true_value
        pos = random.randint(0, len(word))
        return word[:pos] + list(vowel_symbols.values())[np.random.randint(0,len(vowel_symbols.values()))] + \
           word[pos:] if len(word) > 1 and pos < len(word) else word

    def deletion(self) -> str:
        word = self.true_value
        pos = random.randint(0, len(word))
        return word[:pos] + word[pos + 1:] if len(word) > 1 and pos < len(word) else word

    def jumble(self) -> str:
        word = self.true_value
        jumble_type = random.choice([JumbleType.BEG, JumbleType.MID, JumbleType.END])

        return word[0] + ''.join(random.sample(word[1:-1], len(word[1:-1]))) +  \
            word[-1] if jumble_type == JumbleType.MID else \
                (word[0] + ''.join(random.sample(word[1:], len(word[1:]))) if jumble_type == JumbleType.END else 
                ''.join(random.sample(word[:-1], len(word[:-1]))) + word[-1]
                ) if len(word) > 3 else word
        
    def __next__(self):
        if len(self.lst) == 0: 
            return self.true_value
        return random.choice(self.create_erros_gen)()
        

def flattern(dict_items):
    out = []
    for k_i,v_i in dict_items:
        if isinstance(v_i, str):
            out.append(v_i)
        elif isinstance(v_i, list):
            out.extend(v_i)
    return out
                    
class generateErrorsWithDictionary():
    def __init__(self, file_to_add_noise : str, dict_object : dict) -> None:
        self.dict_items = dict_object
        self.lookup = {k : cyclic_iter(flattern(v.items()), k) for k,v in dict_object.items()}
        self.txt = file_to_add_noise

    def addNoiseFile(self, ofile_path) -> None:
        count = 0
        error_count = 0
        with open(self.txt, mode='r', encoding='utf8') as f:
            readlines = f.readlines()
        with open(os.path.join(ofile_path, 'train_06_jun_bad.txt'), mode='w+', encoding='utf8') as f_noisy:
            with open(os.path.join(ofile_path, 'train_06_jun_good.txt'), mode='w+', encoding='utf8') as f_orignal:
                with open(os.path.join(ofile_path, '_log.txt'), mode='w+', encoding='utf8') as f_log:
                    for line in tqdm(readlines): 
                        # tokenized = re.sub('([^\u0D80-\u0DFF\u200a-\u200d]+)', ' \g<1> ', line.strip()).split()
                        tokenized = line.strip().split()
                        for word in tokenized:
                            nword = word
                            count += 1
                            if nword in self.lookup.keys():
                                nword = next(self.lookup[nword])
                                
                            f_noisy.write("{} ".format(nword))
                            f_orignal.write("{} ".format(word))
                            if nword != word:
                                error_count += 1
                                f_log.write(' {} => {} ,'.format(word, nword))
                        
                        f_noisy.write("{} ".format('\n'))
                        f_orignal.write("{} ".format('\n'))
                    f_log.write('\nPercentage {}%'.format((error_count/count)*100))
