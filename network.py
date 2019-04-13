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

        # get length of text, audio
        text_len = text_len.cpu().numpy().astype(np.int32)
        if is_training:
            audio_len = audio_len.cpu().numpy().astype(np.int32)

        # encoder
        output_encoder = self.encoder(text, text_len)

        if is_training:
            mel, align, _, _, _ = self.decoder(audio, output_encoder, text_len, audio_len, mask=True)
        else:
            mel, align, hidden_align, hidden_dec1, hidden_dec2 = self.decoder(audio, output_encoder, text_len)

            for t in range(1, 200):
                last_mel = mel[:,-1,:].unsqueeze(1)

class Encoder(nn.Module):
    def __init__(self, vocab_size, dim_embed, dropout=0.5, num_cbhg=16):
        super(Encoder, self).__init__()
        self.embed = Embedding(vocab_size, dim_embed)
        self.prenet = PreNet(dim_input=dim_embed,
                             dim_hidden=dim_embed,
                             dim_output=dim_embed//2,
                             dropout_rate=dropout)
        self.CBHG = CBHG(K=num_cbhg, dim_input=dim_embed//2, dim_hidden=dim_embed//2)

    def forward(self, text, text_len):
        '''

        :param text: tensor (batch_size, text_size)
        :param text_len: tensor (batch_size)
        :return: embedding tensor (batch_size, text_size, dim_embed)
        '''
        y = self.embed(text)  # (batch_size, text_size, dim_embed)
        y = self.prenet(y) # (batch_size, text_size, dim_embed // 2)
        mask = binaryMask(y, text_len) # mask (batch_size, text_size, dim_embed // 2)
        y = mask * y # (batch_size, text_size, dim_embed // 2)

        y = self.CBHG(y) # (batch_size, text_size, dim_embed)
        mask = binaryMask(y, text_len) # mask (batch_size, text_size, dim_embed)
        output = mask * y # (batch_size, text_size, dim_embed)
        return output

class Decoder(nn.Module):
    def __init__(self, dim_mel, dim_embed, dropout=0.5, reduction_factor=2, is_train=False):
        super(Decoder, self).__init__()
        self.is_train = is_train
        self.prenet = PreNet(dim_input=dim_mel,
                             dim_hidden=dim_embed,
                             dim_output=dim_embed//2,
                             dropout_rate=dropout)
        self.attnRNN = AttentionRNN(dim_input=dim_embed//2,
                                    dim_hidden=dim_embed)
        self.decRNN = DecoderRNN(dim_input = dim_embed + dim_embed,
                                 dim_output=dim_mel,
                                 r=reduction_factor)

    def forward(self, audios, memory, text_length, audio_length, hidden_attn_rnn=None, hidden_dec_rnn=None):
        '''

        :param audios: (batch_size, audio_len, dim_mel)
        :param memory: (batch_size, text_len, dim_embed)
        :param text_length: list text len in batches (batch_size)
        :param audio_length: list audio len in batches (batch_size)
        :param hidden_attn_rnn: it used when eval mode (1, batch_size, dim_embed)
        :param hidden_dec_rnn: it used when eval mode (1, batch_size, dim_embed)
        :return:
        '''
        y = self.prenet(audios) # (batch_size, audio_len, dim_embed // 2)

        if self.is_train:
            # train mode
            y, align, _ = self.attnRNN(y, memory, text_length, hidden_attn_rnn)
            # output (batch_size, audio_len, dim_embed)
            # align (batch_size, audio_len, text_len)

            output, _ = self.decRNN(y, hidden_dec_rnn)
            # output (batch_size, audio_len, dim_audio)
            # align (batch_size, audio_len, text_len)
        else:
            # eval mode
            while True:
                y, align, hidden_attn_rnn = self.attnRNN(y, memory, text_length, hidden_attn_rnn)
                # output (batch_size, 1, dim_embed)
                # align (batch_size, 1, text_len)
                # hidden_attn_rnn (1, batch_size, dim_embed)

                output, hidden_dec_rnn = self.decRNN(y, hidden_dec_rnn)
                # output (batch_size, 1, dim_embed)
                # hidden_dec_rnn (2, batch_size, dim_embed)


        output

