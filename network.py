import torch
import torch.nn as nn
import torch.nn.functional as F
from module import *
from utils.text import *

class Tacotron(nn.Module):
    '''Tacotron'''

    def __init__(self, embed_size):
        super(Tacotron, self).__init__()

        self.vocab_size = get_idx_len()
        self.embed_size = embed_size

        self.encoder = Encoder(vocab_size=self.vocab_size, embed_size=self.embed_size)
        self.decoder = Decoder()

    def forward(self, text, audio, text_len, audio_len, is_training=True):
        text_len = text_len.cpu().numpy().astype(np.int32)

        if is_training:
            audio_len = audio_len.cpu().numpy().astype(np.int32)

        output_encoder = self.encoder(text, text_len)

        if is_training:
            mel, align, _, _, _ = self.decoder(audio, output_encoder, text_len, audio_len, mask=True)
        else:
            mel, align, hidden_align, hidden_dec1, hidden_dec2 = self.decoder(audio, output_encoder, text_len)

            for t in range(1, 200):
                last_mel = mel[:,-1,:].unsqueeze(1)



class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, dropout=0.5, num_chbg=16):
        super(Encoder, self).__init__()
        self.embed = Embedding(vocab_size, embed_size)
        self.prenet = PreNet(input_size=embed_size,
                             hidden_size=embed_size,
                             output_size=embed_size//2,
                             dropout_rate=dropout)
        self.CHBG = CBHG(K=num_chbg, input_size=embed_size//2, hidden_size=embed_size//2)

    def forward(self, input, text_len):
        y = self.embed(input)
        y = self.prenet(y)
        mask = binaryMask(y, text_len)
        y = mask * y

        y, _ = self.CHBG(y)
        mask = binaryMask(y, text_len)
        output = mask * y
        return output

class Decoder(nn.Module):
    def __init__(self, audio_size, embed_size, dropout=0.5, reduction_factor=3):
        super(Decoder, self).__init__()
        self.prenet = PreNet(input_size=audio_size,
                             hidden_size=embed_size,
                             output_size=embed_size//2,
                             dropout_rate=dropout)
        self.attnRNN = AttentionRNN(input_size=embed_size//2,
                                    hidden_size=embed_size,
                                    text_embed_size=embed_size)
        self.decRNN = DecoderRNN(input_size=embed_size+embed_size,
                                 output_size=audio_size,
                                 r=reduction_factor)

    def forward(self, audios, memory, text_length, audio_length):
        '''

        :param audios: (batch_size, audio_len, mel_size)
        :param memory: (batch_size, text_len, embed_size)
        :param text_length: list text len in batches
        :param audio_length: list audio len in batches
        :return:
        '''
        y = self.prenet(audios) # (batch_size, audio_len, ebed_size // 2)
        y, align, hidden = self.attnRNN(y, memory, text_length)
        output

