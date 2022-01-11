from criteria.base_loss import BaseLoss
from criteria.focal_loss import FocalLoss
import torch

class FocalLossL1(BaseLoss):

    def __init__(self, weights=torch.Tensor([0.002, 0.01, 0.0003, 0.04, 0.07, 0.11, 0.16, 0.22, 0.29, 0.0977 ]), gamma=2, device=torch.device("cuda:2")):
        self.weights = weights
        self.gamma = gamma
        super(FocalLossL1, self).__init__()
        self.set_device(device)

    def set_device(self, device):
        self.device = device
        self.weights = self.weights.to(self.device)
        self.focal_loss = FocalLoss(alpha=self.weights, gamma=self.gamma, reduction='mean')

    def forward(self, output, target, weight_mask, mask):
        output = output.to(self.device)
        target = target.to(self.device)
        mask = mask.to(self.device)
        output = torch.mul(output, mask)
        target = torch.mul(target, mask)
        target = target.to(self.device, dtype=torch.long)
        loss = self.focal_loss(output, target)
        return loss