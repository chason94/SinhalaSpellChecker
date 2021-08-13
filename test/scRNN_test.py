import unittest
from scRNN.error_word_generation import AddNoise

class scRNNTest(unittest.TestCase):

    def setUp(self):
        pass
    def test_addNoise(self):
        ofile_path = 'dataset/test.txt'
        AddNoise_obj = AddNoise.getAddNoiseObj(ofile_path)
        AddNoise_obj.addNoiseFile('dataset/')


if __name__ == '__main__':
    unittest.main()