import re
import typing
def read_sentences(file_name):
    with open(file_name, mode='r', encoding='utf8') as f:
        senteces = f.readlines()
    return senteces


def get_words(data_in : typing.Union[str, list], pattern ='[\s]+'):
    if isinstance(data_in, str):
        with open(data_in, mode='r', encoding='utf8') as f:
            words =  re.split(pattern , f.read())
        return words

    elif isinstance(data_in, list):
        copra = []
        for senetence in data_in:
            copra.append(re.split(pattern, senetence.strip()))

        return copra
    else:
        raise NotImplemented


def add_space(data_in : list):
    copra = []
    for senetence in data_in:
        cleaned = re.sub(r'([(",0-9]?)([\u0D80-\u0DFF\u200d]+)([)",0-9]?)', r'\g<1> \g<2> \g<3>', senetence)
        copra.append(cleaned)

    return copra


def clean_numbers_eng_words(data_in : list):
    copra = []
    for senetence in data_in:
        cleaned = re.sub(r'(([:,.-]?)([0-9]+)([,.?]?))+', r'@', senetence)
        cleaned = re.sub(r'([A-z^(<unk>)]+)', r'<unk>', cleaned)
        copra.append(cleaned)

    return copra

    