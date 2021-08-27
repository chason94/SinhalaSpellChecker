__version__ = "0.1.0"
__author__ = 'Charana Sonnadara'
__email__ = "csonnadara02@gmail.com"

from . import seq_modeling
from .corrector_cnnlstm import CnnlstmChecker
from .corrector_lstmlstm import NestedlstmChecker
from .corrector_sclstm import SclstmChecker
from .util import is_module_available

__all__ = []
__all__.extend(["seq_modeling"])


__all__checkers = [
    "CnnlstmChecker",
    "NestedlstmChecker",
    "SclstmChecker",
]

__all__.extend(__all__checkers)


def available_checkers():
    return __all__checkers


class CheckersFactory:
    NAME_TO_CHECKER_MAPPINGS = {
        "cnn-lstm-probwordnoise": CnnlstmChecker,
        "lstm-lstm-probwordnoise": NestedlstmChecker,
        "scrnn-probwordnoise": SclstmChecker,
    }


    @staticmethod
    def from_pretrained(name_or_path, **kwargs):

        import os
        if os.path.exists(name_or_path):
            # name = os.path.split(name_or_path)[-1]
            msg = "To load a model from a path, directy use checker name as XxxChecker instead of using CheckersFactory"
            raise NotImplementedError(msg)

        # create appropriate corrector
        try:
            kwargs.update({"name": name_or_path})
            checker = CheckersFactory.NAME_TO_CHECKER_MAPPINGS[name_or_path](**kwargs)
            checker.from_pretrained()
            return checker
        except KeyError as e:
            msg = f"Found checker name: {name_or_path}. " \
                  f"Expected a checker name in {[*CheckersFactory.NAME_TO_CHECKER_MAPPINGS.keys()]}"
            raise Exception(msg) from e


__all__.extend(["CheckersFactory"])
