import torch
import torch.nn as nn
from criteria.base_loss import BaseLoss

class Raw3dOrdinalRegressionLoss(BaseLoss):
    """
    Ordinal loss is defined as the average of pixelwise ordinal loss F(h, w, X, O)
    over the entire image domain:
    """

    def __init__(self, device=torch.device("cuda:0")):
        super(Raw3dOrdinalRegressionLoss, self).__init__(device)

    def forward(self, ord_labels, target):

        N, C, H, W = ord_labels.size()
        ord_num = C

        loss = 0.0

        K = torch.zeros((N, C, H, W), dtype=torch.int)
        for i in range(ord_num):
            K[:, i, :, :] = K[:, i, :, :] + i * torch.ones((N, H, W), dtype=torch.int)

        mask_0 = (K <= target).detach()
        mask_1 = (K > target).detach()

        one = torch.ones(ord_labels[mask_1].size())
        # if torch.cuda.is_available():
        #     one = one.cuda()

        loss += torch.sum(torch.log(torch.clamp(ord_labels[mask_0], min=1e-8, max=1e8))) \
                     + torch.sum(torch.log(torch.clamp(one - ord_labels[mask_1], min=1e-8, max=1e8)))

        # del K
        # del one
        # del mask_0
        # del mask_1

        N = N * H * W
        loss /= (-N)  # negative
        return loss