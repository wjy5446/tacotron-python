import os

import torch
import torch.nn as nn
import torch.optim as optim

import seaborn as sns

from network import Tacotron
from dataset import TTS_Dataset
from loss import TacoLoss


def train():
    epochs = 1
    batch_size = 4

    # load dataset
    dataset = TTS_Dataset('data', batch_size=4)

    ## model
    model = Tacotron(dim_embed=256, dim_mel=80, dim_mag=1025, is_training=True)

    # loss
    criterion = TacoLoss()

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.99))

    total_batch = len(dataset)
    for epoch in range(epochs):
        for idx, data in enumerate(dataset):
            model.train()
            output_mel, output_mag, align = model(data['text'], data['mel'], data['text_len'], data['mel_len'])
            loss = criterion(output_mel, data['mel'], output_mag, data['mag'])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if ((epoch * total_batch) + idx) % 10 == 0:
                print('Epoch [%d/%d], Iter [%d/%d], loss: %.4f'
                      % (epoch, epochs, idx + 1, total_batch, loss.item()))

                save_att_result('result', align[0].detach().numpy(), epoch, idx, loss.item())

def save_att_result(path, align, epoch, iter, loss):
    print(align.shape)
    ax = sns.heatmap(align)
    fig = ax.get_figure()
    name_save = 'epoch-{}_iter-{}_loss-{}.png'.format(epoch, iter, loss)
    print(fig)
    #fig.savefig(os.path.join(path, name_save))

    print('[INFO] Save attention heatmap!!')

if __name__ == "__main__":
    train()