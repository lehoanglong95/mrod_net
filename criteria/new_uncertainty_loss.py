# import torch
# from torch.nn import Parameter
# from criteria.base_loss import BaseLoss
# from utils.utils import import_file, convert_str_from_underscore_to_camel
# import torch.nn as nn
#
# class NewUncertaintyLoss(BaseLoss):
#
#     def __init__(self, loss_files, device=torch.device("cuda:1")):
#         super(NewUncertaintyLoss, self).__init__()
#         self.device = device
#         self.losses = [self.__load_loss(file_name) for file_name in loss_files]
#         self.params = Parameter(torch.empty(len(self.losses)).to(self.device))
#         nn.init.uniform_(self.params, 0.2, 1)
#
#     def __load_loss(self, file_name):
#         if "/" in file_name:
#             cls_name = convert_str_from_underscore_to_camel(file_name.split("/")[-1])
#         else:
#             cls_name = convert_str_from_underscore_to_camel(file_name)
#         return getattr(import_file(file_name), cls_name)()
#
#     def set_device(self, device):
#         self.device = device
#         for loss in self.losses:
#             loss.set_device(device)
#
#     def forward(self, predicts, targets, weight_mask, mask):
#         loss = 0
#         for idx, (param, loss_func, predict, target) in enumerate(zip(self.params, self.losses, predicts, targets)):
#             # idx: 0 L2 Loss, idx > 0: Ordinal Loss
#             if idx == 0:
#                 precision = 2 * (param ** 2)
#                 loss += 1 / precision * loss_func(predict, target, weight_mask, mask) + torch.log(param)
#             else:
#                 precision = param ** 2
#                 loss += 1 / precision * loss_func(predict, target, weight_mask, mask) + torch.log(param)
#         return loss
import torch
from torch.nn import Parameter
from criteria.base_loss import BaseLoss
from utils.utils import import_file, convert_str_from_underscore_to_camel
import torch.nn as nn

class NewUncertaintyLoss(BaseLoss):

    def __init__(self, loss_weights, loss_files, device=torch.device("cuda:1")):
        super(NewUncertaintyLoss, self).__init__()
        self.loss_weights = loss_weights
        self.losses = [self.__load_loss(file_name) for file_name in loss_files]
        # if type(self.device) is list:
        #     self.params = Parameter(torch.empty(len(self.losses), device=self.device[len(self.device) - 1]))
        # else:
        #     self.params = Parameter(torch.empty(len(self.losses), device=self.device))
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
        if type(self.device) is list:
            # self.params = Parameter(torch.empty(len(self.losses), device=self.device[len(self.device) - 1]))
            # nn.init.uniform_(self.params, 0.2, 1)
            if len(self.device) < len(self.losses):
                for loss in self.losses:
                    loss.set_device(device)
            else:
                for d, loss in zip(self.device, self.losses):
                    loss.set_device(d)
        else:
            # self.params = Parameter(torch.empty(len(self.losses), device=self.device))
            # nn.init.uniform_(self.params, 0.2, 1)
            for loss in self.losses:
                loss.set_device(device)

    def forward(self, predicts, targets, weight_mask, mask):
        loss = 0
        for idx, (loss_weight, loss_func, predict, target) in enumerate(zip(self.loss_weights, self.losses, predicts, targets)):
            if type(self.device) is list:
                weight_mask = weight_mask.to(self.device[idx])
                mask = mask.to(self.device[idx])
                target = target.to(self.device[idx])
            # loss += 1 / (param ** 2) * loss_func(predict, target, weight_mask, mask) + torch.log(param ** 2)
            loss += loss_weight * loss_func(predict, target, weight_mask, mask)
        return loss