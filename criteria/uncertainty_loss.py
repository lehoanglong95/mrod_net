import torch
from torch.nn import Parameter
from criteria.base_loss import BaseLoss
from utils.utils import import_file, convert_str_from_underscore_to_camel
import torch.nn as nn

class UncertaintyLoss(BaseLoss):

    def __init__(self, loss_files, device=torch.device("cuda:1")):
        super(UncertaintyLoss, self).__init__()
        # self.device = device
        self.losses = [self.__load_loss(file_name) for file_name in loss_files]
        self.params = Parameter(torch.zeros(len(self.losses), dtype=torch.float32), requires_grad=True).to(device)
        # nn.init.uniform_(self.params, 0.2, 1)
        self.set_device(device)

    def __load_loss(self, file_name):
        if "/" in file_name:
            cls_name = convert_str_from_underscore_to_camel(file_name.split("/")[-1])
        else:
            cls_name = convert_str_from_underscore_to_camel(file_name)
        return getattr(import_file(file_name), cls_name)()

    def set_device(self, device):
        self.device = device
        for loss in self.losses:
            loss.set_device(device)

    def forward(self, predicts, targets, weight_mask, mask):
        loss = 0
        for param, loss_func, predict, target in zip(self.params, self.losses, predicts, targets):
            loss += torch.exp(-param) * loss_func(predict, target, weight_mask, mask) + torch.log(1 + torch.exp(param))
            # loss += 1 / (param ** 2) * loss_func(predict, target, weight_mask, mask) + torch.log(param ** 2)
        return loss