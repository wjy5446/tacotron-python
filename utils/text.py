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
    return text

if __name__=='__main__':
    tokenize('안')