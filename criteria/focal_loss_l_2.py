from criteria.base_loss import BaseLoss
from criteria.focal_loss import FocalLoss
import torch

class FocalLossL2(BaseLoss):

    def __init__(self, weights=torch.Tensor([0.009, 0.56, 0.25, 0.10, 0.033, 0.007, 0.002, 0.002, 0.0001, 0.045]), gamma=2, device=torch.device("cuda:3")):
        self.weights = weights
        self.gamma = gamma
        super(FocalLossL2, self).__init__()
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