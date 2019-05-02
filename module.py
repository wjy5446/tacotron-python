import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import Attention
from utils.text import *

###########
# Encoder Module
###########

class CBHG(nn.Module):
    def __init__(self, K, dim_input, dim_hidden, dim_proj_hiddens):
        super(CBHG, self).__init__()

        ## conv1d bank
        layers_conv1d_bank = []

        for k in range(1, K+1):
            layers_conv1d_bank += [
                nn.Sequential(
                    nn.Conv1d(dim_input, dim_hidden, kernel_size=k, padding=k//2),
                    nn.ReLU(),
                    nn.BatchNorm1d(dim_hidden, momentum=0.99, eps=1e-3)
                )
            ]

        self.conv1d_bank = layers_conv1d_bank

        ## maxpooling
        self.max1d = nn.MaxPool1d(2, 1, 1)

        ## conv1d_projection
        self.conv1d_projection = nn.Sequential(
            nn.Conv1d(K * dim_hidden, dim_proj_hiddens[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(dim_proj_hiddens[0], momentum=0.99, eps=1e-3),
            nn.Conv1d(dim_proj_hiddens[0], dim_proj_hiddens[1], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(dim_proj_hiddens[1], momentum=0.99, eps=1e-3),
        )
        self.fc_projection = nn.Linear(dim_proj_hiddens[1], dim_hidden)

        ## highway network
        layers_highway = []
        for k in range(4):
            layers_highway += [HighwayNet(dim_input=dim_hidden,
                                          dim_output=dim_hidden)]
        self.highway = nn.Sequential(*layers_highway)

        ## GRU
        self.gru = nn.GRU(input_size=dim_hidden,
                          hidden_size=dim_hidden,
                          bidirectional=True,
                          batch_first=True)

    def forward(self, input):
        '''

        :param input: (batch_size, text_size, dim_input)
        :return:
        '''
        x = input.transpose(1, 2) # (batch_size, dim_input, text_size)
        seq_time = x.size(-1)
        ## conv1d_banks
        y = torch.cat([conv1d(x)[:,:,:seq_time] for conv1d in self.conv1d_bank], dim=1) # (batch_size, K * dim_hidden, text_size)

        ## maxpooling
        y = self.max1d(y)[:,:,:seq_time] # (batch_size, K * dim_hidden, text_size)

        ## conv1d projections
        y = self.conv1d_projection(y)  # (batch_size, dim_hidden, text_size)
        y = y + x
        y = y.transpose(1, 2) # (batch_size, text_size, dim_hidden)

        ## highway
        y = self.fc_projection(y)
        y = self.highway(y)  # (batch_size, text_size, dim_hidden)

        ## gru
        output, _ = self.gru(y) # (batch_size, text_size, dim_hidden * 2)

        return output


class Embedding(nn.Module):
    def __init__(self, vocab_size, dim_embed):
        """Embedding"""
        super(Embedding, self).__init__()
        self.net = nn.Embedding(vocab_size, dim_embed, padding_idx=char2id['_'])

    def forward(self, input):
        '''
        :param input: (batch_size, text_size)
        :return: embedding (batch_size, text_size, dim_embed)
        '''
        embed = self.net(input) # (batch_sie, text_size, dim_embed)
        return embed

class PreNet(nn.Module):
    """Prenet"""

    def __init__(self, dim_input, dim_hidden, dim_output, dropout_rate):
        super(PreNet, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim_hidden, dim_output),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

    def forward(self, input):
        '''
        prenet
        :param input: (batch_size, seq_size, dim_input)
        :return: (batch_size, seq_size, dim_output)
        '''
        return self.net(input) # (batch_size, seq_size, dim_output)

class HighwayNet(nn.Module):
    """HighwayNet"""
    def __init__(self, dim_input, dim_output):
        super(HighwayNet, self).__init__()
        self.H = nn.Linear(dim_input, dim_output)
        self.T = nn.Linear(dim_input, dim_output)

    def forward(self, input):
        '''
        Highway Net
        :param input: (batch_size, seq_size, dim_input)
        :return: (batch_size, seq_size, dim_output
        '''
        h = F.relu(self.H(input))
        t = torch.sigmoid(self.T(input))

        return h * t + input * (1. - t) # (batch_size, seq_size, dim_output)

###########
# Decoder Module
###########

def binaryMask(x, length):
    '''

    :param x: (batch_size, seq_len, dim)
    :param length: (batch_size)
    :return: mask (batch_size, seq_len, dim)
    '''

    batch_size, seq_len, dim = x.size()
    mask_batch = []
    for b in range(batch_size):
        mask = []
        len = length[b]
        for l in range(seq_len):
            if l < len:
                mask.append(np.ones((dim)))
            else:
                mask.append(np.zeros((dim)))
        mask_batch.append(np.stack(mask))

    mask_batch = np.stack(mask_batch)
    mask_batch = torch.from_numpy(mask_batch).to(x.device).type(x.dtype) # (batch_size, seq_len, dim)

    return mask_batch


class AttentionRNN(nn.Module):
    def __init__(self, dim_input, dim_hidden):
        super(AttentionRNN, self).__init__()
        self.gru = nn.GRU(input_size=dim_input,
                          hidden_size=dim_hidden,
                          batch_first=True)
        self.attention = Attention(query_size=dim_hidden,
                                   context_size=dim_hidden)

    def forward(self, input, memory, text_length, hidden_attn_rnn=None):
        '''

        :param input: (batch_size, audio_len, dim_audio),
                    if train mode then audio_len is full size
                    if eval mode then audio_len is 1
        :param memory: (batch_size, text_len, dim_embed)
        :param text_length: text_len list in batch (batch_size)
        :param hidden_attn_rnn: eval (1, batch_size, dim_hidden)
        :return: context + hidden (batch_size, audio_len, hidden_dim * 2)
        '''
        is_train = hidden_attn_rnn is None
        dim_hidden = memory.size(2)
        audio_len = input.size(1)

        self.gru.flatten_parameters()

        if is_train:
            output, hidden = self.gru(input)
        else:
            output, hidden = self.gru(input, hidden_attn_rnn)
        # output (batch_size, audio_len, dim_hidden)
        # hidden (1, batch_size, dim_hidden)

        mask_attention = self.makeAttnMask(memory, text_length, audio_len) # mask (batch_size, audio_len, text_len)
        align = self.attention(query=output, memory=memory, memory_mask=mask_attention) # align (batch_size, audio_len, text_len)
        align_tiled = align.unsqueeze(3).repeat(1, 1, 1, dim_hidden) # (batch_size, audio_len, text_len, dim_hidden)
        memory_tiled = memory.unsqueeze(1).repeat(1, input.size(1), 1, 1) # (batch_size, audio_len, text_len, dim_hidden)
        context = torch.sum(align_tiled * memory_tiled, dim=2) # (batch_size, audio_len, dim_hidden)
        output = torch.cat([context, output], dim=2) # (batch_size, audio_len, dim_hidden * 2)
        # output (batch_size, audio_len, dim_hidden * 2)
        # align (batch_size, audio_len, text_len)
        # hidden (1, batch_size, dim_hidden)
        return output, align, hidden

    def makeAttnMask(self, memory, text_len, audio_len):
        '''
        :param memory: (batch_size, text_len, dim_hidden)
        :param text_len: text_len list in batch (batch_size)
        :return: Attention Mask (batch_size, audio_len, text_len)
        '''


        batch_size, length, _ = memory.size()
        mask_batch = []

        for b in range(batch_size):
            mask = []
            len = text_len[b]
            for l in range(length):
                if l < len:
                    mask.append(np.ones((audio_len), dtype=np.int32))
                else:
                    mask.append(np.zeros((audio_len), dtype=np.int32))

            mask = np.stack(mask)
            mask_batch.append(mask)

        mask_batch = np.stack(mask_batch)
        mask_batch = np.swapaxes(mask_batch, 1, 2)
        mask_batch = torch.from_numpy(mask_batch).type(torch.ByteTensor).to(memory.device) # (batch_size, audio_len, text_len)
        return mask_batch

class DecoderRNN(nn.Module):
    def __init__(self, dim_input, dim_output, r=2):
        super(DecoderRNN, self).__init__()
        self.r = r
        self.dim_output = dim_output
        self.gru1 = nn.GRU(input_size=dim_input,
                           hidden_size=dim_input, batch_first=True)
        self.gru2 = nn.GRU(input_size=dim_input,
                           hidden_size=dim_input, batch_first=True)
        self.fc = nn.Linear(dim_input, r * dim_output)

    def forward(self, input, hidden_dec_rnn=None):
        '''

        :param input: (batch_size, audio_len, dim_input)
                    if train mode then audio_len is full size
                    if eval mode then audio_len is 1
        :param hidden_dec_rnn: (2, batch_size, dim_input)
        :return:
        output : (batch_size, r * audio_len, dim_output)
        hidden : (2, batch_size, dim_input)
        '''
        is_train = hidden_dec_rnn is None
        batch_size = input.size(0)

        if is_train:
            output1, hidden1 = self.gru1(input)
            # output1 (batch_size, audio_len, dim_input)
            # hidden1 (1, batch_size, dim_input)

            output2, hidden2 = self.gru2(input + output1)
            # output2 (batch_size, audio_len, dim_input)
            # hidden2 (1, batch_size, dim_input)
        else:
            hidden_dec_rnn1 = hidden_dec_rnn[0, :, :] # (1, batch_size, dim_input)
            hidden_dec_rnn2 = hidden_dec_rnn[1, :, :] # (1, batch_size, dim_input)

            output1, hidden1 = self.gru1(input, hidden_dec_rnn1)
            # output1 (batch_size, 1, dim_input)
            # hidden1 (1, batch_size, dim_input)

            output2, hidden2 = self.gru2(input + output1, hidden_dec_rnn2)
            # output2 (batch_size, 1, dim_input)
            # hidden2 (1, batch_size, dim_input)

        output = self.fc(output1 + output2 + input) # output (batch_size, audio_len, r * dim_output)
        output = output.view(batch_size, -1, self.dim_output) # (batch_size, audio_len * r, dim_output)

        return output, torch.cat([hidden1, hidden2], dim=0)

