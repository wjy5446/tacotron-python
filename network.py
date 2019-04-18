import torch
import torch.nn as nn
import torch.nn.functional as F
from module import *
from utils.text import *
from dataset import TTS_Dataset

class Tacotron(nn.Module):
    '''Tacotron'''

    def __init__(self, dim_embed, dim_mel, dim_mag, reduction_factor=2, is_training=False):
        super(Tacotron, self).__init__()

        self.vocab_size = get_idx_len()
        self.dim_embed = dim_embed
        self.dim_mel = dim_mel
        self.dim_mag = dim_mag

        self.encoder = Encoder(vocab_size=self.vocab_size, dim_embed=self.dim_embed)
        self.decoder = Decoder(dim_mel, dim_embed, is_training=is_training)
        self.postprocessing = PostProcessingNet(dim_input=dim_mel, dim_hidden=dim_embed,
                                                dim_mel=dim_mel, dim_mag=dim_mag)

    def forward(self, text, audio, text_len, audio_len, is_training=True):

        # get length of text, audio
        text_len = text_len.cpu().numpy().astype(np.int32)
        if is_training:
            audio_len = audio_len.cpu().numpy().astype(np.int32)
        else:
            audio_len = None

        # encoder
        output_encoder = self.encoder(text, text_len) # (batch_size, text_len, dim_embed)

        # decoder
        output_mel, align = self.decoder(audio, output_encoder, text_len, audio_len) # (batch_size, r * audio_len, dim_mel)

        # postprecessing net
        output_mag = self.postprocessing(output_mel) # (batch_size, r * audio_len, dim_mag)
        return output_mel, output_mag, align

#################################
# Encoder
#################################

class Encoder(nn.Module):
    def __init__(self, vocab_size, dim_embed, dropout=0.5, num_cbhg=16):
        super(Encoder, self).__init__()
        self.embed = Embedding(vocab_size, dim_embed)
        self.prenet = PreNet(dim_input=dim_embed,
                             dim_hidden=dim_embed,
                             dim_output=dim_embed//2,
                             dropout_rate=dropout)
        self.CBHG = CBHG(K=num_cbhg,
                         dim_input=dim_embed//2,
                         dim_hidden=dim_embed//2,
                         dim_proj_hiddens=[dim_embed//2, dim_embed//2])

    def forward(self, input, text_len):
        '''

        :param input: tensor (batch_size, text_size)
        :param text_len: tensor (batch_size)
        :return: embedding tensor (batch_size, text_size, dim_embed)
        '''
        y = self.embed(input)  # (batch_size, text_size, dim_embed)
        y = self.prenet(y) # (batch_size, text_size, dim_embed // 2)
        mask = binaryMask(y, text_len) # mask (batch_size, text_size, dim_embed // 2)
        y = mask * y # (batch_size, text_size, dim_embed // 2)

        y = self.CBHG(y) # (batch_size, text_size, dim_embed)
        mask = binaryMask(y, text_len) # mask (batch_size, text_size, dim_embed)
        output = mask * y # (batch_size, text_size, dim_embed)
        return output

#################################
# Decoder
#################################

class Decoder(nn.Module):
    def __init__(self, dim_mel, dim_embed, dropout=0.5, reduction_factor=2, is_training=False):
        super(Decoder, self).__init__()
        self.is_training = is_training
        self.dim_mel = dim_mel
        self.dim_embed = dim_embed
        self.reduction_factor = reduction_factor
        self.max_audio_len = 300

        self.prenet = PreNet(dim_input=dim_mel,
                             dim_hidden=dim_embed,
                             dim_output=dim_embed//2,
                             dropout_rate=dropout)
        self.attnRNN = AttentionRNN(dim_input=dim_embed//2,
                                    dim_hidden=dim_embed)
        self.decRNN = DecoderRNN(dim_input = dim_embed + dim_embed,
                                 dim_output=dim_mel,
                                 r=reduction_factor)

    def forward(self, audios, memory, text_length, audio_length):
        '''

        :param audios: (batch_size, audio_len, dim_mel)
        :param memory: (batch_size, text_len, dim_embed)
        :param text_length: list text len in batches (batch_size)
        :param audio_length: list audio len in batches (batch_size)
        :return: output # (batch_size, r * audio_len, dim_embed)
        '''

        batch_size = memory.size(0)

        # init go frame, hidden_rnn
        go_frame = torch.zeros((batch_size, 1, self.dim_mel))

        if self.is_training:
            mask = binaryMask(audios, audio_length)
            y = mask * audios
            y = torch.cat([go_frame, y[:,self.reduction_factor::self.reduction_factor,:].float()], dim=1)
            # (batch_size, audio_len // r, dim_mel)
            y = self.prenet(y) # (batch_size, audio_len // r, dim_embed // 2)

            # train mode
            y, align, _ = self.attnRNN(y, memory, text_length)
            # output (batch_size, audio_len // r, dim_embed)
            # align (batch_size, audio_len // r, text_len)

            output, _ = self.decRNN(y)
            # output (batch_size, r * (audio_len), dim_embed)
            # align (batch_size, audio_len, text_len)
        else:
            # eval mode
            y = go_frame
            hidden_attn_rnn = torch.zeros((1, batch_size, self.dim_embed))
            hidden_dec_rnn = torch.zeros((2, batch_size, self.dim_embed))

            li_output = []

            while True:
                y, align, hidden_attn_rnn = self.attnRNN(y, memory, text_length, hidden_attn_rnn)
                # output (batch_size, 1, dim_embed)
                # align (batch_size, 1, text_len)
                # hidden_attn_rnn (1, batch_size, dim_embed)

                output, hidden_dec_rnn = self.decRNN(y, hidden_dec_rnn)
                # output (batch_size, r, dim_embed)
                # hidden_dec_rnn (2, batch_size, dim_embed)

                y = output[:,-1,:]
                li_output.append(output)

                # stop condition
                if len(li_output) > self.max_audio_len:
                    break

            output = torch.cat(li_output, dim=1) # (batch_size, r * audio_len, dim_embed

        return output, align

#################################
# Post Processing Net
#################################

class PostProcessingNet(nn.Module):
    '''Post processing Net'''
    def __init__(self, dim_input, dim_hidden, dim_mel, dim_mag, num_cbhg=8):
        super(PostProcessingNet, self).__init__()
        self.cbhg = CBHG(K=num_cbhg,
                         dim_input=dim_input,
                         dim_hidden=dim_hidden//2,
                         dim_proj_hiddens=[dim_hidden, dim_mel])
        self.fc = nn.Linear(dim_hidden, dim_mag)
    def forward(self, input):
        '''
        Post precessing Net
        :param input: (batch_size, r * audio_len , dim_embed)
        :return: (batch_size, r * audio_len, dim_embed)
        '''

        y = self.cbhg(input) # (batch_size, r * audio_len, dim_embed)
        y = self.fc(y) # (batch_size, r * audio_len, dim_mag)
        return y

if __name__ == '__main__':
    vocab_size = get_idx_len()

    dataset = TTS_Dataset('data')
    info = dataset.__getitem__(0)
    print('text : ', info['text'].size(), 'mel : ', info['mel'].size(), 'mag : ', info['mag'].size())
    print('text_len : ', info['text_len'], 'mel_len : ', info['mel_len'])

    model = Tacotron(dim_embed=256, dim_mel=80, dim_mag=1025, is_training=True)
    output_mel, output_mag = model(info['text'], info['mel'], info['text_len'], info['mel_len'])
    print(output_mel.size(), output_mag.size())