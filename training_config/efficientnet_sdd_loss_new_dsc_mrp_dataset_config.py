import torchvision.transforms as transforms
from data_augmentation.center_crop import CenterCrop
from data_augmentation.horizontal_flip import HorizontalFlip
import torch
from training_config.base_new_dsc_mrp_config import BaseNewDscMrpConfig
from constants import *


class _Config(BaseNewDscMrpConfig):
    #'/home/longlh/Desktop', '/home/longlh/Downloads/patient_list.xlsx'
    def __init__(self, old_root_dir="/home/longlh/hard_2/Workspace",
                 new_root_dir="/data1/long/longlh/longlh/Workspace",
                 csv_file="/data1/long/longlh/longlh/densenet_implementation/dataset/split_non_overlap_dsc_mrp_dataset.csv"):
        super(_Config, self).__init__(old_root_dir, new_root_dir, csv_file)
        self.model_parallel = True
        self.network_architecture = {
            "file": "network_architecture/efficientnet_backbone_drnn_sdd_loss",
            "parameters": {
                "n_channels": 60,
                "growth_rate": 12,
                "reduction": 0.5,
                "k_ordinal_class": 5,
                "dev_0": torch.device("cuda:0"),
                "dev_1": torch.device("cuda:0"),
                "dev_2": torch.device("cuda:1"),
                "dev_3": torch.device("cuda:1"),
                "model_name": "efficientnet-b0"
            }
        }
        self.loss = {
            "loss1": {
                "file": "criteria/test_loss",
                "parameters": {
                    "loss_files": ["criteria/average_phase_loss", "criteria/ordinal_regression_loss"],
                    "device": torch.device("cuda:1")
                }
            }
        }
        self.loss_weights = [1]
        self.val_loss = {
            "loss1": {
                "file": "criteria/average_phase_loss",
                "parameters": {
                    "device": torch.device("cuda:1")
                }
            }
        }
        setattr(self, DatasetTypeString.TRAIN, {
            "dataset": {
                "file": "new_dsc_mrp_dataset",
                "parameters": {
                    "old_root_dir": f"{old_root_dir}",
                    "new_root_dir": f"{new_root_dir}",
                    "csv_file": f"{csv_file}",
                    "dataset_type": DatasetType.TRAIN,
                    "file_names": {'inputs': 'IMG_n01.npy',
                                   'labels': 'phase_maps_medfilt_rs_n.npy',
                                   'ord_decreasing_labels_5_classes': 'phase_maps_or_spacing_decreasing_discretization_label_medfilt_rs_5classes_n.npy',
                                   'labels_weight_for_each_phase': 'phase_maps_medfilt_rs_n_wm.npy',
                                   'mask': 'mask_4d.npy'},
                    "transform": transforms.Compose([CenterCrop(224), HorizontalFlip(0.5)])
                }
            },
            "generator": {
                "batch_size": 1,
                "shuffle": True,
                "num_workers": 8
            }
        })


config = _Config().__dict__
