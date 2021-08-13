import typing
import re
from tqdm import tqdm




sinhala_unicode = {
    "ං"  : '\u0D82', 
    "ඃ"  : '\u0D83', 
    "අ"    : '\u0D85', 
    "ආ"   : '\u0D86', 
    "ඇ"   : '\u0D87', 
    "ඈ"   : '\u0D88', 
    "ඉ"   : '\u0D89', 
    "ඊ"   : '\u0D8A', 
    "උ"   : '\u0D8B', 
    "ඌ"   : '\u0D8C', 
    "ඍ"   : '\u0D8D', 
    "ඎ"   : '\u0D8E',
    "ඏ"   : '\u0D8F',
    "ඐ"   : '\u0D90', 
    "එ"   : '\u0D91',
    "ඒ"   : '\u0D92',
    "ඓ"   : '\u0D93',
    "ඔ"   : '\u0D94',
    "ඕ"   : '\u0D95',
    "ඖ"   : '\u0D96',
    "ක"   : '\u0D9A',
    "ඛ"   : '\u0D9B',
    "ග"   : '\u0D9C',
    "ඝ"   : '\u0D9D',
    "ඞ"   : '\u0D9E',
    "ඟ"   : '\u0D9F',
    "ච"   : '\u0DA0',
    "ඡ"   : '\u0DA1',
    "ජ"   : '\u0DA2',
    "ඣ"   : '\u0DA3',
    "ඤ"   : '\u0DA4',
    "ඥ"   : '\u0DA5',
    "ඦ"   : '\u0DA6',
    "ට"   : '\u0DA7',
    "ඨ"   : '\u0DA8',
    "ඩ"   : '\u0DA9',
    "ඪ"   : '\u0DAA',
    "ණ"   : '\u0DAB',
    "ඬ"   : '\u0DAC',
    "ත"   : '\u0DAD',
    "ථ"   : '\u0DAE',
    "ද"   : '\u0DAF',
    "ධ"   : '\u0DB0',
    "න"   : '\u0DB1',
    "ඳ"   : '\u0DB3',
    "ප"   : '\u0DB4',
    "ඵ"   : '\u0DB5',
    "බ"   : '\u0DB6',
    "භ"   : '\u0DB7',
    "ම"   : '\u0DB8',
    "ඹ"   : '\u0DB9',
    "ය"   : '\u0DBA',
    "ර"   : '\u0DBB',
    "ල"   : '\u0DBD',
    "ව"   : '\u0DC0',
    "ශ"   : '\u0DC1',
    "ෂ"   : '\u0DC2',
    "ස"   : '\u0DC3',
    "හ"   : '\u0DC4',
    "ළ"   : '\u0DC5',
    "ෆ"   : '\u0DC6',
    "්"   : '\u0DCA',
    "ා"   : '\u0DCF',
    "ැ"   : '\u0DD0',
    "ෑ"   : '\u0DD1',
    "ි"   : '\u0DD2',
    "ී"   : '\u0DD3',
    "ු"   : '\u0DD4',
    "ූ"   : '\u0DD6',
    "ෘ"   : '\u0DD8',
    "ෙ"   : '\u0DD9',
    "ේ"   : '\u0DDA',
    "ෛ"   : '\u0DDB',
    "ො"   : '\u0DDC',
    "ෝ"   : '\u0DDD',
    "ෞ"   : '\u0DDE',
    "ෟ"   : '\u0DDF',
    # "෦"   : '\u0DE6',
    # "෧"   : '\u0DE7',
    # "෨"   : '\u0DE8',
    # "෩"   : '\u0DE9',
    # "෪"   : '\u0DEA',
    # "෫"   : '\u0DEB',
    # "෬"   : '\u0DEC',
    # "෭"   : '\u0DED',
    # "෮"   : '\u0DEE',
    # "෯"   : '\u0DEF',
    "ෲ"   : '\u0DF2',
    "ෳ"   : '\u0DF3',
    # "෴"   : '\u0DF4',
}

vowel_symbols = {
    "ං"  : '\u0D82', 
    "ඃ"  : '\u0D83', 
    "්"   : '\u0DCA',
    "ා"   : '\u0DCF',
    "ැ"   : '\u0DD0',
    "ෑ"   : '\u0DD1',
    "ි"   : '\u0DD2',
    "ී"   : '\u0DD3',
    "ු"   : '\u0DD4',
    "ූ"   : '\u0DD6',
    "ෙ"   : '\u0DD9',
    "ේ"   : '\u0DDA',
    "ෛ"   : '\u0DDB',
    "ො"   : '\u0DDC',
    "ෝ"   : '\u0DDD',
    "ෞ"   : '\u0DDE',
    "ෟ"   : '\u0DDF',
    "ෘ"   : '\u0DD8',
    "ෲ"   : '\u0DF2',
    "ෳ"   : '\u0DF3',
    "‍්‍ය" : '\u0DCA\u200D\u0DBA', 
    "‍්‍ර" : '\u0DCA\u200D\u0DBB',
    "ZWJ" : '\u200d' ,  # ZERO WIDTH JOINER'
    "ZWNJ" :'\u200c' ,  # ZERO WIDTH NON-JOINER (U+200C)
    "ZWS" :'\u200b'  # ZEROS WITH SPACE
}


common_erros_corrections = {
    # from : to
    # 'අා' to 'ආ'
    '`ද' : "ඳ",
    '`ඩ' : "ඬ",
    '`ඩ' : "ඬ",
    "`ග" : "ඟ",
    "`ඵ" : "ළු",
    "ඡුා" : "ඡා",
    'අා' : "ආ",
    # 'අැ', 'ඇ'
    'අැ' :  'ඇ',
    # අෑ ඈ
    'අෑ' :  'ඈ',
    # ඒ ඒ
    'ඒ' :  'ඒ',
    # 'ෙ', 'ච', 'ෙ'
    '\s(\u0DD9)([\u0D99-\u0DC6])(\u0DD9)' : '\g<2>\u0DDB',
    # 'ැ', 'ු'
    '\u0DD0\u0DD4' : '\u0DD0',
    # උෟ, ඌ
    'උෟ': 'ඌ',
    # unwanted ZWJ
    # '\s([\u200d\u200b\u200c]+)' : '',
    # '([\u200d\u200b\u200c]+)\s' : '',
    #  "ෛ",
    '\u0DD9\u0DD9' : vowel_symbols["ෛ"],
    #  "ේ"
    '\u0DD9\u0DCA' : vowel_symbols["ේ"],
    #  "ො"   
    '\u0DD9\u0DCF' : vowel_symbols["ො"],
    #  "ෝ" 
    '\u0DD9\u0DCF\u0DCA' : vowel_symbols["ෝ"],
    '\u0DD9\u0DCA\u0DCF' : vowel_symbols["ෝ"],
    '\u0DDC\u0DCA' : vowel_symbols["ෝ"],
    # "ෞ",
    '\u0DD9\u0DDF' : vowel_symbols["ෞ"],
    # "ෲ",
    '\u0DD8\u0DD8' : vowel_symbols["ෲ"],
}

encode_error_correction_rules = {
    '([\u0D9A-\u0DBD\u0DC0-\u0DC6])\s([\u0DCA\u0DCF\u0DD0-\u0DD6\u0DD9-\u0DDF\u0DD8\u0DF2\u0DF3]|(\u0DCA\u200D\u0DBA)|(\u0DCA\u200D\u0DBB))' : '\g<1>\g<2>',
    # '([^\u0D80-\u0DFF])\s([\u0DCA\u0DCF\u0DD0-\u0DD6\u0DD9-\u0DDF\u0DD8\u0DF2\u0DF3]|(\u0DCA\u200D\u0DBA)|(\u0DCA\u200D\u0DBB))' : '\g<1>',
    # "([\u0DCA\u0DCF\u0DD0-\u0DD6\u0DD9-\u0DDF\u0DD8\u0DF2\u0DF3])\s([\u0DCA\u0DCF\u0DD0-\u0DD6\u0DD9-\u0DDF\u0DD8\u0DF2\u0DF3])\s((\u0DCA\u200D\u0DBA)|(\u0DCA\u200D\u0DBB))" :"\g<1>\g<2>\g<3>"
    
}

def unicode_error_correction(str_in:str, 
                            debug = False) -> str:


    cleaned = str_in
    for error, corr in common_erros_corrections.items():
        cleaned = re.sub(error, corr, cleaned)

    for _,v in vowel_symbols.items():
        reg = r"{" + v + "}{2,}"
        cleaned = re.sub(reg, v, cleaned)

    if str_in != cleaned and debug:
        for word, result in zip(str_in.split(), cleaned.split()):
            if word != result:
                print("unicode_error_correction : word {} raw {} cleaned {}".format(word, list(str_in), list(cleaned)))

    return cleaned


def unicode_error_viewer(word):
    # combine multiple charachters to represent one
    for error, corr in common_erros_corrections.items():
        errors = re.findall(error, word)
        if len(errors):
            print("unicode_error_viewer : {}".format(error))
    
    # multiple charachters
    for vi,v in vowel_symbols.items():
        reg = r"{" + v + "}{2,}"
        errors = re.findall(reg, word)
        if len(errors):
            print("unicode_error_viewer : {}".format(vi))
    # multipe vowel symbols are togther
    reg = '['+''.join(
        ['({})'.format(vsym) for vsym in vowel_symbols.keys() 
        if vsym not in ["ZWJ", "ZWNJ", "ZWS", "‍්‍ය", "‍්‍ර"]])+']{2,}'
    errors = re.findall(reg, word)
    if len(errors):
        # print(errors)
        print("unicode_error_viewer {} : {}".format(word, "multiple adjacent vowels :"), list(word))


def correct_words(text : str, src_target : dict, debug = False) -> str:
    cleaned = text
    for src, target in src_target.items():
        cleaned_0 = re.sub('([^\u0D80-\u0DFF\u200a-\u200d])(' + src + ')([^\u0D80-\u0DFF\u200a-\u200d])','\g<1>' + target + '\g<3>' , cleaned)
        if cleaned_0!= cleaned and debug:
            print(src, target)
        cleaned = cleaned_0

    return cleaned

def encode_error_correction(str_in : str,
                            debug = False):

    cleaned = str_in
    for error, corr in encode_error_correction_rules.items():
        cleaned = re.sub(error, corr, cleaned)
    if debug:
        print(str_in, list(str_in, cleaned, list(str_in)))
    return cleaned