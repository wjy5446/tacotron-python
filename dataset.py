import os
import json
import numpy as np

import torch
from torch.utils.data import Dataset

from utils.text import *
from utils.audio import *

class TTS_Dataset(Dataset):
    def __init__(self, root, batch_size=32):
        self.root = root
        self.batch_size = batch_size

        li_wave, li_text = self.load_info(root)
        self.n_data = len(li_wave)
        self.waves = li_wave
        self.texts = li_text
        self.texts = self.preprocess_text(self.texts)

        self.sort()

        item = self.make_data(self.waves[1], self.texts[1])
        print(item)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        if idx + self.batch_size <= self.n_data:
            idx_batch = range(idx, idx + self.batch_size)
        else:
            idx_batch = range(self.n_data - self.batch_size, self.n_data)

        li_batch = [self.make_data(self.waves[j], self.texts[j]) for j in idx_batch]
        item = make_data

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
        mel, mag = get_spectrogram(wav, sampling_rate=22050)

        return {'text': text, 'mel': mel, 'mag': mag}

def make_data_batch(li_batch):



if __name__ == '__main__':
    dataset = TTS_Dataset('data')
