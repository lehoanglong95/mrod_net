import pytorch_ssim
from criteria.base_loss import BaseLoss
import torch

class SsimLoss(BaseLoss):

    def __init__(self, device=torch.device("cuda:0")):
        super(SsimLoss, self).__init__(device)
        self.ssim_loss = pytorch_ssim.SSIM()

    def forward(self, predict, target, weight_mask, mask, use_weight_mask=True):
        loss = 0
        predict_with_mask = torch.mul(predict, mask.repeat(1, 5, 1, 1, 1))
        target_with_mask = torch.mul(target, mask.repeat(1, 5, 1, 1, 1))
        n, c, d, w, h = predict_with_mask.shape
        for i in range(n):
            for j in range(c):
                for k in range(d):
                    predict_img = predict_with_mask[i][j][k]
                    target_img = target_with_mask[i][j][k]
                    loss += (1 - self.ssim_loss(predict_img, target_img))
        loss /= (n * c * d)
        return loss