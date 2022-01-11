from criteria.base_loss import BaseLoss
from criteria.focal_loss import FocalLoss
import torch

class FocalLossL(BaseLoss):

    def __init__(self, weights=None, gamma=None, device=torch.device("cuda:0")):
        super(FocalLossL, self).__init__()
        self.set_device(device)
        weights = weights.to(self.device)
        self.focal_loss = FocalLoss(alpha=weights, gamma=gamma, reduction='mean')

    def forward(self, output, target, weight_mask, mask):
        output = output.to(self.device)
        target = target.to(self.device)
        mask = mask.to(self.device)
        output = torch.mul(output, mask)
        target = torch.mul(target, mask)
        target = target.to(self.device, dtype=torch.long)
        loss = self.focal_loss(output, target)
        return loss