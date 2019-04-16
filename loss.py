import torch.nn as nn

class TacoLoss(nn.Module):
    def __init__(self, weight_mel=1, weight_mag=1):
        super(TacoLoss, self).__init__()
        self.weight_mel = weight_mel
        self.weight_mag = weight_mag

        self.loss = nn.L1Loss()

    def __call__(self, pred_mel, true_mel, pred_mag, true_mag):
        loss_mel = self.loss(pred_mel, true_mel) * self.weight_mel
        loss_mag = self.loss(pred_mag, true_mag) * self.weight_mag

        return loss_mel + loss_mag