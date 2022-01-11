import torch
from torch.utils import data
import numpy as np
import torchvision.transforms as transforms
import argparse
from constants import DatasetType
from dsc_mrp_dataset_for_generate import DscMrpDataset
import copy

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = vars(parser.parse_args())
    # phases = ['ColArt', 'ColCap', 'ColEVen', 'ColLVen', 'ColDel']
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    epochs = 190
    params = {"batch_size": 1,
              "shuffle": False,
              "num_workers": 4}
    transform = transforms.Compose([transforms.Resize((224, 244)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                         std=[0.5, 0.5, 0.5])])
    training_dataset = DscMrpDataset('/media/data1/longlh', '/home/quiiubuntu/longlh/patient_list.xlsx', DatasetType.VAL,
                                     {"ord_label": "phase_maps_or_spacing_increasing_discretization_label_medfilt_rs_n.npy",
                                      "mask": "mask_4d.npy"})
    training_generator = data.DataLoader(training_dataset, **params)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    for batch_idx, (ord_label, mask, dir_name) in enumerate(training_generator):

        # print(mask.shape)
        # C, D, W, H = mask.shape
        # mask = mask.view(1, C, D, W, H)
        ord_label = ord_label[0]
        mask = mask[0]
        ord_label = ord_label.numpy()
        mask = mask.numpy()
        phase_l = []
        beta = 0.9999
        real_ord_label = ord_label[np.tile(mask,(5, 1, 1, 1)) > 0]
        temp_wm = [np.sum(real_ord_label == 0), np.sum(real_ord_label == 1), np.sum(real_ord_label == 2), np.sum(real_ord_label == 3)
                    ,np.sum(real_ord_label == 4)]
        ord_label_cp = copy.deepcopy(ord_label)
        ord_label_cp[ord_label_cp == 0] = (1 - beta) / (1 - beta ** temp_wm[0])
        ord_label_cp[ord_label_cp == 1] = (1 - beta) / (1 - beta ** temp_wm[1])
        ord_label_cp[ord_label_cp == 2] = (1 - beta) / (1 - beta ** temp_wm[2])
        ord_label_cp[ord_label_cp == 3] = (1 - beta) / (1 - beta ** temp_wm[3])
        ord_label_cp[ord_label_cp == 4] = (1 - beta) / (1 - beta ** temp_wm[4])
        # print(ord_label_cp.shape)
        # for i in range(5):
        #     phase = phase_map[0][i]
        #     real_phase_map = phase[mask[0][0] > 0]
        #     temp_weight_map = np.histogram(real_phase_map, bins)
        #     temp_weight_map[0][temp_weight_map[0] == 0] = 1
        #     # invert log to generate wm
        #     # weight_map_tuple = (np.log(1 / (temp_weight_map[0] / real_phase_map.shape[0])), temp_weight_map[1])
        #     # class balanced_loss_based_on_effective_number_of_samples with beta = 0.9999
        #     weight_map_tuple = ((1 - beta) / (1 - beta ** temp_weight_map[0]), temp_weight_map[1])
        #     for idx in range(len(weight_map_tuple[0])):
        #         if idx < len(weight_map_tuple[0]) - 1:
        #             phase[(phase >= weight_map_tuple[1][idx]) & (phase < weight_map_tuple[1][idx + 1])] = \
        #             weight_map_tuple[0][idx]
        #         else:
        #             phase[(phase >= weight_map_tuple[1][idx]) & (phase <= weight_map_tuple[1][idx + 1])] = \
        #             weight_map_tuple[0][idx]
        #     d, w, h = phase.shape
        #     phase = phase.reshape(1, d, w, h)
        #     phase *= mask[0]
        #     phase_l.append(phase)
        np.save(dir_name[0] + "/" + f"class_balanced_loss_phase_maps_or_spacing_increasing_discretization_label_medfilt_rs_n_beta_{beta}.npy", ord_label_cp)