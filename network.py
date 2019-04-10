import torch
import torch.nn as nn
import torch.nn.functional as F
from module import *

class Tacotron(nn.Module):
    def __init__(self):
        super(Tacotron, self).__init__()

        self.encoder = Encoder(vocab_size=20)
        self.decoder = Decoder()

class Encoder(nn.Module):

    def __init__(self, vocab_size, embed_size):
        super(Encoder, self).__init__()
        self.embed = Embedding(vocab_size)
        self.prenet = PreNet(input_size=256,
                             hidden_size=256,
                             output_size=128,
                             dropout_rate=0.5)
        self.CHBG = CBHG(K=16, input_size=128, hidden_size=128)

    def forward(self, input, text_length):
        y = self.embed(input)
        y = self.prenet(y)
        mask = binaryMask(y, text_length)
        y = mask * y

        y, _ = self.CHBG(y)
        mask = binaryMask(y, text_length)
        output = mask * y
        return output

class Decoder(nn.Module):
    def __init__(self, text_embed_size, reduction_factor=2):
        super(Decoder, self).__init__()
        self.text_embed_size = text_embed_size
        self.prenet = PreNet(input_size=256,
                             hidden_size=256,
                             output_size=128,
                             dropout_rate=0.5)
        self.attnRNN = AttentionRNN(input_size=128,
                                    hidden_size=256,
                                    text_embed_size=text_embed_size)
        self.decRNN = DecoderRNN(input_size=256+text_embed_size,
                                 output_size=70,
                                 r=2)

    def forward(self, frames, memory, text_length):
        y = self.prenet(frames)
        y, align, hidden = self.attnRNN(y, memory, text_length)
        output

