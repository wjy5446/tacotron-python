import torch
import torch.nn as nn
import torch.optim as optim

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

            if (idx + 1) % 1000 == 0:
                print('Epoch [%d/%d], Iter [%d/%d], loss: %.4f'
                      % (epoch, epochs, idx + 1, total_batch, loss.item()))

                print(align)

            break

#
if __name__ == "__main__":
    train()