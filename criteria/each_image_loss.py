import torch
from criteria.base_loss import BaseLoss

class EachImageLoss(BaseLoss):

    def __init__(self, device=torch.device("cuda:0")):
        super(EachImageLoss, self).__init__()
        self.device = device
        self.loss = 0.0

    # input size: N, C, D, W, H
    def forward(self, predict, target, weight_mask, mask):
        self.loss = 0
        predict_with_mask = torch.mul(predict, mask.repeat(1, 5, 1, 1, 1))
        target_with_mask = torch.mul(target, mask.repeat(1, 5, 1, 1, 1))
        absolute_err = target_with_mask - predict_with_mask
        matrix_loss = torch.mul(torch.mul(absolute_err, absolute_err), weight_mask)
        n, c, d, w, h = matrix_loss.shape
        for i in range(n):
            for j in range(c):
                for k in range(d):
                    elements_counts = matrix_loss[i][j][k][mask[0][0][j] > 0].shape[0]
                    self.loss += torch.sum(matrix_loss[i][j][k]) / elements_counts
        self.loss /= d
        self.loss /= c
        return self.loss