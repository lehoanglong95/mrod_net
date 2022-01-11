import torch
from torch.utils import data
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import argparse
board_writer = SummaryWriter()

class DatasetType:
    TRAIN = 0
    VAL = 1
    TEST = 2

class RAWDSCMRPDataset(data.Dataset):

    def __init__(self, root_dir, excel_file, dataset_type, phase_maps, mask, transform=None):
        idx_array = self._get_index_array(excel_file, dataset_type)
        self.phase_maps, self.masks, self.dir_names = self._get_file_names(excel_file, idx_array, root_dir, phase_maps, mask)
        # assert len(self.inputs) == len(self.labels) == len(self.labels_weight) == len(self.mask_brain)

    def _get_index_array(self, excel_file, dataset_type, sheet_name='train_val_test_idx'):
        train_val_test_idx = pd.read_excel(excel_file, sheet_name=sheet_name, header=None)
        train_val_test_idx = train_val_test_idx.rename(columns={0: 'type'})
        return train_val_test_idx.loc[train_val_test_idx['type'] == dataset_type].index.values.astype(int)

    def _get_file_names(self, excel_file, idx_array, root_dir, phase_maps, mask, sheet_name='filenames'):
        file_names_df = pd.read_excel(excel_file, sheet_name=sheet_name, header=None)
        file_names_df = file_names_df.rename(columns={0: 'dir'})
        phase_maps_l = []
        masks = []
        dir_names = []
        for i in idx_array:
            file_name_l = file_names_df['dir'][i].split('/')[2:-1] # remove root dir
            phase_maps_l.append(root_dir + "/" + "Workspace" + "/" + "/".join(file_name_l) + "/" + phase_maps)
            masks.append(root_dir + "/" + "Workspace" + "/" + "/".join(file_name_l) + "/" + mask)
            dir_names.append(root_dir + "/" + "Workspace" + "/" + "/".join(file_name_l))
        return phase_maps_l, masks, dir_names

    def __len__(self):
        return len(self.phase_maps)

    def __getitem__(self, item):
        # input_image = np.load(self.inputs[item])
        # phase_1 = np.load(self.phase_1_name[item])
        # phase_2 = np.load(self.phase_2_name[item])
        # phase_3 = np.load(self.phase_3_name[item])
        # phase_4 = np.load(self.phase_4_name[item])
        # phase_5 = np.load(self.phase_5_name[item])
        # label = np.load(self.labels[item])
        # label_weight = np.load(self.labels_weight[item])
        # mask_brain = np.load(self.mask_brain[item])
        return (np.load(self.phase_maps[item]), np.load(self.masks[item]), self.dir_names[item])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', "--bins", type=int, default=None)
    args = vars(parser.parse_args())
    bins = args['bins']
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
    # training_dataset = RAWDSCMRPDataset('/media/data1/longlh', '/home/quiiubuntu/longlh/patient_list.xlsx', DatasetType.TRAIN, 'phase_maps_medfilt_rs_n.npy', 'mask_4d.npy')
    training_dataset = RAWDSCMRPDataset('/home/longlh/hard_2', '/home/longlh/Downloads/patient_list.xlsx',
                                        DatasetType.TRAIN, 'phase_maps_medfilt_rs_n.npy', 'mask_4d.npy')
    training_generator = data.DataLoader(training_dataset, **params)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    for batch_idx, (phase_map, mask, dir_name) in enumerate(training_generator):

        # print(mask.shape)
        # C, D, W, H = mask.shape
        # mask = mask.view(1, C, D, W, H)
        phase_map = phase_map.numpy()
        mask = mask.numpy()
        phase_l = []
        beta = 0.9999
        for i in range(5):
            phase = phase_map[0][i]
            real_phase_map = phase[mask[0][0] > 0]
            temp_weight_map = np.histogram(real_phase_map, bins)
            temp_weight_map[0][temp_weight_map[0] == 0] = 1
            # invert log to generate wm
            # weight_map_tuple = (np.log(1 / (temp_weight_map[0] / real_phase_map.shape[0])), temp_weight_map[1])
            # class balanced_loss_based_on_effective_number_of_samples with beta = 0.9999
            effective_num = 1.0 - np.power(beta, temp_weight_map[0])
            weights = (1.0 - beta) / np.array(effective_num)
            weights = weights / np.sum(weights) * bins * 10
            weight_map_tuple = (weights, temp_weight_map[1])
            d, w, h = phase.shape
            wm_phase = np.zeros((d, w, h))
            for idx in range(len(weight_map_tuple[0])):
                if idx < len(weight_map_tuple[0]) - 1:
                    wm_phase[(phase >= weight_map_tuple[1][idx]) & (phase < weight_map_tuple[1][idx + 1])] = \
                    weight_map_tuple[0][idx]
                else:
                    wm_phase[(phase >= weight_map_tuple[1][idx]) & (phase <= weight_map_tuple[1][idx + 1])] = \
                    weight_map_tuple[0][idx]
            d, w, h = wm_phase.shape
            wm_phase = wm_phase.reshape(1, d, w, h)
            wm_phase *= mask[0]
            phase_l.append(wm_phase)
        # print(torch.cat(phase_l).shape)
        # real_phase_map = phase_map[mask.repeat(1, 5, 1, 1, 1) > 0].numpy()
        # temp_weight_map = np.histogram(real_phase_map, 50)
        # temp_weight_map[0][temp_weight_map[0] == 0] = 1
        # weight_map_tuple = (np.log(1 / (temp_weight_map[0] / real_phase_map.shape[0])), temp_weight_map[1])
        # for idx in range(len(weight_map_tuple[0])):
        #     if idx < len(weight_map_tuple[0]) - 1:
        #         phase_map[(phase_map >= weight_map_tuple[1][idx]) & (phase_map < weight_map_tuple[1][idx + 1])] = weight_map_tuple[0][idx]
        #     else:
        #         phase_map[(phase_map >= weight_map_tuple[1][idx]) & (phase_map <= weight_map_tuple[1][idx + 1])] = weight_map_tuple[0][idx]
        np.save(dir_name[0] + "/" + f"class_balanced_loss_bins_{bins}_beta_{beta}_wm.npy", np.concatenate(phase_l))
