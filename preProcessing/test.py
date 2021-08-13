import unittest
from error_correction import unicode_error_correction, unicode_error_viewer

class TestStringMethods(unittest.TestCase):

    def setUp(self):
        self.word_0 = ["ගෞරවාදරයට", "පැවතිණි", "වුව", "ඇ","ප", "පුරෝගාමියා"]
        self.word_1 =[' ෙමම',' ෙරගුලාසි',' ෙමම','ෛවද්‍', 'වෙෙද්‍ය', 'ස්ෛවරිභාවය', 'ස්ෙෙවරීභාවය'
'ෙම්ජර්', 'ෙප්‍රේමා', 'ෙනො',' ෙතොරතුරු',' ෙමොණරාගල', ' ෙතා', 'මාතලේ්', 'ආසියාන',
'ා', '්', 'වෛද්ය', 'වෛද්යවරු', 'අවශ්ය', 'සම්පේ්‍රෂණ', 'ක්රියාකාරකම්', 'ප්රමාණය'
'රාජකාා', 'ශාස්ත්‍ර්‍ර්‍ර්‍ර්‍ර්‍ර්‍රාලය']
    def test_error_correction(self):
        
        for word in self.word_0+self.word_1:
            unicode_error_correction(word, True)
        
    def test_error_viewer(self):
        for word in self.word_0+self.word_1:
            unicode_error_viewer(word)
        



if __name__ == '__main__':
    unittest.main()