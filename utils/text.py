import re
from jamo import hangul_to_jamo, h2j

PAD = '_'
EOS = '~'
PUNC = '!\'(),-.:;?'
SPACE = ' '

JAMO_LEADS = "".join([chr(_) for _ in range(0x1100, 0x1113)])
JAMO_VOWELS = "".join([chr(_) for _ in range(0x1161, 0x1176)])
JAMO_TAILS = "".join([chr(_) for _ in range(0x11A8, 0x11C3)])

VALID_CHARS = JAMO_LEADS + JAMO_VOWELS + JAMO_TAILS + PUNC + SPACE
ALL_SYMBOLS = PAD + EOS + VALID_CHARS

char2id = {c: i for i, c in enumerate(ALL_SYMBOLS)}
id2char = {i: c for i, c in enumerate(ALL_SYMBOLS)}

NUMBER2HAN = {
    '0': '영',
    '1': '하나',
    '2': '둘',
    '3': '셋',
    '4': '넷',
    '5': '다섯',
    '6': '여섯',
    '7': '일곱',
    '8': '여덟',
    '9': '아홉',
}

ENG2HAN = {
    'a': '에이', 'b': '비', 'c': '씨',
    'd': '디', 'e': '이', 'f': '에프',
    'g': '쥐', 'h': '에이치', 'i': '아이',
    'j': 'ㅓ', 'k': '케이', 'l': '엘',
    'm': '엠', 'n': '엔', 'o': '오',
    'p': '피', 'q': '큐', 'r': '알',
    's': '에스', 't': '티', 'u': '유',
    'v': '브이', 'w': '더블유', 'x': '엑스',
    'y': '와', 'z': '제트'
}

SPE2HAN = {
    '%': '퍼센트',
    '$': '달러'
}


def sent2idx(sentence):
    idx = [char2id[word] for word in sentence]
    return idx

def get_idx_len():
    return len(char2id)

def tokenize(text):
    text = normalize(text)
    tokens = list(hangul_to_jamo(text))

    return tokens + ['~']

def normalize(text):
    text = text.strip()
    
    # change number to hangul
    li_number = re.findall('[0-9]',text)
    for num in li_number:
        text = re.sub(num, NUMBER2HAN[num], text)
    
    # change eng to hangul
    li_eng = re.findall('[a-zA-Z]',text)
    for eng in li_eng:
        text = re.sub(eng, ENG2HAN[eng.lower()], text)
    
    # change spe to hangul
    li_spe= re.findall('[%$]',text)
    for spe in li_spe:
        text = re.sub(spe, SPE2HAN[spe], text)    
    
    # remove punc data
    text = re.sub('[“""]', '', text)
    return text

if __name__=='__main__':
    print(tokenize('"안녕2" A하세요3'))