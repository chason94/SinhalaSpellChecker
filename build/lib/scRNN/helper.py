import re
import bisect
def decode_word(X, calc_argmax, id2vocab):
    if calc_argmax:
        X = X.argmax(axis=-1)
    return ' '.join(id2vocab[x] for x in X)

def colors(token, color='green'):
  colors = {
      'green' :'\033[92m',  # green
      'red' :'\033[91m',  # red
      # 'close' :'\033[0m',  # close
    }

  return colors[color] + token + '\033[0m'


def decode_word(X, calc_argmax, id2vocab):
    if calc_argmax:
        X = X.argmax(axis=-1)
    return ' '.join(id2vocab[x] for x in X)


def safe_division(n, d):
    return n / d if d > 0 else 0

def get_words(filename):
    with open(filename, 'r+', encoding='utf8') as f:
      return list(set([x.strip() for x in f.read().split() if not bool(re.match(r'[A-z0-9]',x))]))

class WeightedTuple(object):
    def __init__(self, items):
        self.indexes = []
        self.items = []
        next_index = 0
        for key in items.keys():
            val = items[key]
            self.indexes.append(next_index)
            self.items.append(key)
            next_index += val

        self.len = next_index

    def __getitem__(self, n):
        if n < 0:
            n = self.len + n
        if n < 0 or n >= self.len:
            raise IndexError

        idx = bisect.bisect_right(self.indexes, n)
        return self.items[idx-1]

    def __len__(self):
        return self.len



