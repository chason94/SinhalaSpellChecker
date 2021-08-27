from typing import List

from .commons import spacy_tokenizer
from .corrector import Corrector
from .seq_modeling.cnnlstm import load_model, load_pretrained, model_predictions, model_inference
from .seq_modeling.helpers import load_data

""" corrector module """


class CnnlstmChecker(Corrector):

    def load_model(self, ckpt_path):
        print(f"initializing model")
        initialized_model = load_model(self.vocab)
        self.model = load_pretrained(initialized_model, self.ckpt_path, device=self.device)

    def correct_strings(self, mystrings: List[str], return_all=False) -> List[str]:
        self.is_model_ready()
        if self.tokenize:
            mystrings = [spacy_tokenizer(my_str) for my_str in mystrings]
        data = [(line, line) for line in mystrings]
        batch_size = 4 if self.device == "cpu" else 4
        return_strings = model_predictions(self.model, data, self.vocab, device=self.device, batch_size=batch_size)
        if return_all:
            return mystrings, return_strings
        else:
            return return_strings

