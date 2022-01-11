from constants import *
from training_config.base_config import BaseConfig
from data_augmentation.center_crop import CenterCrop
from data_augmentation.horizontal_flip import HorizontalFlip
import torchvision.transforms as transforms

class BaseNewDscMrpConfig(BaseConfig):

    def __init__(self, old_root_dir, new_root_dir, csv_file):
        super(BaseNewDscMrpConfig, self).__init__()
        self.epochs = 300
        # self.val_loss = {
        #     "loss1": {
        #         "file": "criteria/average_phase_loss"
        #     }
        # }
        self.val_loss = None
        self.val_loss_weights = [1]
        self.loss_weights = [1]
        self.validation_leap = 4
        self.optimizer = {
            "name": OptimizerType.ADAM,
            "parameters": {
                "init_setup": {
                    "lr": 0.001,
                    "betas": (0.9, 0.999,),
                    "eps": 10 ** -8
                }
            }
        }
        self.lr_scheduler = {
            "name": OptimizerLrScheduler.ReduceLROnPlateau,
            "parameters": {
                "factor": 0.75,
                "patience": 3
            }
        }
        setattr(self, DatasetTypeString.VAL, {
            "dataset": {
                "file": "new_dsc_mrp_dataset",
                "parameters": {
                    "old_root_dir": f"{old_root_dir}",
                    "new_root_dir": f"{new_root_dir}",
                    "csv_file": f"{csv_file}",
                    "dataset_type": DatasetType.VAL,
                    "file_names": {'inputs': 'IMG_n01.npy',
                                   'labels': 'phase_maps_medfilt_rs_n.npy',
                                   'labels_weight': 'phase_maps_medfilt_rs_n_wm.npy',
                                   'mask': 'mask_4d.npy'},
                    "transform": transforms.Compose([CenterCrop(224), HorizontalFlip(0.5)])
                }
            },
            "generator": {
                "batch_size": 1,
                "shuffle": True,
                "num_workers": 5
            }
        })
        setattr(self, DatasetTypeString.TEST, {
            "dataset": {
                "file": "new_dsc_mrp_dataset",
                "parameters": {
                    "old_root_dir": f"{old_root_dir}",
                    "new_root_dir": f"{new_root_dir}",
                    "csv_file": f"{csv_file}",
                    "dataset_type": DatasetType.TEST,
                    "file_names": {'inputs': 'IMG_n01.npy',
                                   'labels': 'phase_maps_medfilt_rs_n.npy',
                                   'mask': 'mask_4d.npy'},
                    "transform": transforms.Compose([CenterCrop(224)])
                }
            },
            "generator": {
                "batch_size": 1,
                "shuffle": False,
                "num_workers": 5
            }
        })


