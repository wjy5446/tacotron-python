import os
import json
import numpy as np

import torch
from torch.utils.data import Dataset

from utils.text import *
from utils.audio import *

class TTS_Dataset(Dataset):
    def __init__(self, root, batch_size=4):
        self.root = root
        self.batch_size = batch_size

        li_wave, li_text = self.load_info(root)
        print('[INFO] Load data from {}'.format(root))

        self.n_data = len(li_wave)
        self.waves = li_wave
        self.texts = li_text

        self.texts = self.preprocess_text(self.texts)
        print('[INFO] Preprocess the text (text -> index)')

        self.sort()
        print('[INFO] Sort the data by length of sentence')
        print('[INFO] Data size : {}'.format(len(self.texts)))

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        if idx + self.batch_size <= self.n_data:
            idx_batch = range(idx, idx + self.batch_size)
        else:
            idx_batch = range(self.n_data - self.batch_size, self.n_data)

        li_batch = [self.make_data(self.waves[j], self.texts[j]) for j in idx_batch]
        item = make_data_batch(li_batch)

        return item


    def load_info(self, root):
        path_json = os.path.join(root, 'info.json')
        with open(path_json, 'r') as f:
            info = json.load(f)

        li_wave = list(info.keys())
        li_text = list(info.values())

        return li_wave, li_text

    def preprocess_text(self, texts):
        texts_new = []
        for text in texts:
            text = normalize(text)
            text = tokenize(text)
            texts_new.append(text)

        return texts_new

    def sort(self):
        li_len_text = [len(self.texts[i]) for i in range(self.n_data)]

        # sort
        idx_sort = sorted(range(self.n_data), key=lambda i: -li_len_text[i])
        self.waves = [self.waves[i] for i in idx_sort]
        self.texts = [self.texts[i] for i in idx_sort]

        return None

    def make_data(self, wave, text):
        path_wave = os.path.join(self.root, wave)
        wav, _ = load_audio(path_wave, sample_rate=22050)
        mag, mel = get_spectrogram(wav, sampling_rate=22050)

        return {'text': text, 'mel': mel, 'mag': mag}

def make_data_batch(li_batch):
    text = [sent2idx(batch['text']) for batch in li_batch]
    mel = [batch['mel'] for batch in li_batch]
    mag = [batch['mag'] for batch in li_batch]

    max_text_len = max([len(x) for x in text])
    max_audio_len = max([x.shape[0] for x in mel])

    # 3배수의 길이로 audio 생성
    remain_audio = max_audio_len % 2
    max_audio_len += 2 - remain_audio

    li_text_len = []
    li_audio_len = []

    # 나머지 공간에 padding(-) 을 추가
    for i, t in enumerate(text):
        text_len = len(t)
        len_padding = max_text_len - text_len
        padding = sent2idx(['_' for _ in range(len_padding)])
        text[i] += padding
        li_text_len.append(text_len)

    #  나머지 공간을 0으로 패딩
    for i, m in enumerate(mel):
        mel_len = m.shape[0]
        len_padding = max_audio_len - mel_len
        padding = np.zeros([len_padding, m.shape[1]])
        mel[i] = np.concatenate([m, padding], axis=0)
        li_audio_len.append(mel_len)

    # 나머지 공간을 0으로 패딩
    for i, m in enumerate(mag):
        mag_len = m.shape[0]
        len_padding = max_audio_len - mag_len
        padding = np.zeros([len_padding, m.shape[1]])
        mag[i] = np.concatenate([m, padding], axis=0)

    return {'text': torch.LongTensor(text),
            'mel': torch.Tensor(mel),
            'mag': torch.Tensor(mag),
            'text_len': torch.LongTensor(li_text_len),
            'mel_len': torch.LongTensor(li_audio_len)}

if __name__ == '__main__':
    dataset = TTS_Dataset('data')
    info = dataset.__getitem__(0)
    print('text : ', info['text'].size(), 'mel : ', info['mel'].size(), 'mag : ', info['mag'].size())
    print('text_len : ', info['text_len'], 'mel_len : ', info['mel_len'])



