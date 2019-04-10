from jamo import hangul_to_jamo, h2j

def tokenize(text):
    text = normalize(text)
    tokens = list(hangul_to_jamo(text))

    return tokens + ['EOS']

def normalize(text):
    text = text.strip()

    return text

if __name__=='__main__':
    tokenize('안')