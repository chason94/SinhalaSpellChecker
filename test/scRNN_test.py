import unittest
from scRNN.error_word_generation import AddNoise
from preProcessing.error_correction import unicode_error_correction

class scRNNTest(unittest.TestCase):

    def setUp(self):
        pass
    def test_addNoise(self):
        ofile_path = 'dataset/test.txt'
        AddNoise_obj = AddNoise.getAddNoiseObj(ofile_path)
        AddNoise_obj.addNoiseFile('dataset/')


class encodeTesting(unittest.TestCase):
    def setUp(self):
        pass
    def test_encoding(self):
        # "incorrect" : "correct"
        words = {"අා": "ආ",
                 # "එ්": "ඒ",
                 "අැ": "ඇ",
                 }
        for word in words.items():
            self.assertEqual(unicode_error_correction(word[0], debug= False), word[1])



if __name__ == '__main__':
    unittest.main()