import torch
from torch.utils import data
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

board_writer = SummaryWriter()

class DatasetType:
    TRAIN = 0
    VAL = 1
    TEST = 2

class RAWDSCMRPDataset(data.Dataset):

    def __init__(self, root_dir, excel_file, dataset_type, inputs_name, phase_1_name, phase_2_name, phase_3_name, phase_4_name, phase_5_name, transform=None):
        idx_array = self._get_index_array(excel_file, dataset_type)
        self.inputs, self.phase_1_name, self.phase_2_name, self.phase_3_name, self.phase_4_name, self.phase_5_name, self.dir_names = self._get_file_names(excel_file, idx_array, root_dir, inputs_name, phase_1_name, phase_2_name, phase_3_name, phase_4_name, phase_5_name)
        # assert len(self.inputs) == len(self.labels) == len(self.labels_weight) == len(self.mask_brain)

    def _get_index_array(self, excel_file, dataset_type, sheet_name='train_val_test_idx'):
        train_val_test_idx = pd.read_excel(excel_file, sheet_name=sheet_name, header=None)
        train_val_test_idx = train_val_test_idx.rename(columns={0: 'type'})
        return train_val_test_idx.loc[train_val_test_idx['type'] == dataset_type].index.values.astype(int)

    def _get_file_names(self, excel_file, idx_array, root_dir, inputs_name, phase_1_name, phase_2_name, phase_3_name, phase_4_name, phase_5_name, sheet_name='filenames'):
        file_names_df = pd.read_excel(excel_file, sheet_name=sheet_name, header=None)
        file_names_df = file_names_df.rename(columns={0: 'dir'})
        inputs = []
        phase_1_names = []
        phase_2_names = []
        phase_3_names = []
        phase_4_names = []
        phase_5_names = []
        dir_names = []
        for i in idx_array:
            file_name_l = file_names_df['dir'][i].split('/')[2:-1] # remove root dir
            inputs.append(root_dir + "/" + "workspace" + "/" + "/".join(file_name_l) + "/" + inputs_name)
            phase_1_names.append(root_dir + "/" + "workspace" + "/" + "/".join(file_name_l) + "/" + phase_1_name)
            phase_2_names.append(root_dir + "/" + "workspace" + "/" + "/".join(file_name_l) + "/" + phase_2_name)
            phase_3_names.append(root_dir + "/" + "workspace" + "/" + "/".join(file_name_l) + "/" + phase_3_name)
            phase_4_names.append(root_dir + "/" + "workspace" + "/" + "/".join(file_name_l) + "/" + phase_4_name)
            phase_5_names.append(root_dir + "/" + "workspace" + "/" + "/".join(file_name_l) + "/" + phase_5_name)
            dir_names.append(root_dir + "/" + "workspace" + "/" + "/".join(file_name_l))
        return inputs, phase_1_names, phase_2_names, phase_3_names, phase_4_names, phase_5_names, dir_names

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        input_image = np.load(self.inputs[item])
        phase_1 = np.load(self.phase_1_name[item])
        phase_2 = np.load(self.phase_2_name[item])
        phase_3 = np.load(self.phase_3_name[item])
        phase_4 = np.load(self.phase_4_name[item])
        phase_5 = np.load(self.phase_5_name[item])
        # label = np.load(self.labels[item])
        # label_weight = np.load(self.labels_weight[item])
        # mask_brain = np.load(self.mask_brain[item])
        return (input_image, phase_1, phase_2, phase_3, phase_4, phase_5, self.dir_names[item])

if __name__ == '__main__':
    phases = ['ColArt', 'ColCap', 'ColEVen', 'ColLVen', 'ColDel']
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
    # '/media/data1/longlh', '/home/quiiubuntu/longlh/patient_list.xlsx'
    # '/home/longlh/Desktop', '/home/longlh/Downloads/patient_list.xlsx'
    training_dataset = RAWDSCMRPDataset('/home/longlh/Desktop', '/home/longlh/Downloads/patient_list.xlsx', DatasetType.TEST, 'IMG_n01.npy', f'{phases[0]}.npy', f'{phases[1]}.npy', f'{phases[2]}.npy', f'{phases[3]}.npy', f'{phases[4]}.npy')
    training_generator = data.DataLoader(training_dataset, **params)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    for batch_idx, (inputs, phase_1, phase_2, phase_3, phase_4, phase_5, dir_name) in enumerate(training_generator):
        # a = [phase_1[0], phase_2[0], phase_3[0], phase_4[0], phase_5[0]]
        # phase_maps = np.concatenate(a)
        p_1 = phase_1[0]
        p_2 = phase_2[0]
        p_3 = phase_3[0]
        p_4 = phase_4[0]
        p_5 = phase_5[0]
        # rescale_phase_maps = (phase_maps - phase_maps.min()) / (phase_maps.max() - phase_maps.min())
        # normalized_phase = 1.8 * rescale_phase_maps - 0.9
        # rescale from 0 to 1
        rescale_phase1 = (p_1 - p_1.min()) / (p_1.max() - p_1.min())
        rescale_phase2 = (p_2 - p_2.min()) / (p_2.max() - p_2.min())
        rescale_phase3 = (p_3 - p_3.min()) / (p_3.max() - p_3.min())
        rescale_phase4 = (p_4 - p_4.min()) / (p_4.max() - p_4.min())
        rescale_phase5 = (p_5 - p_5.min()) / (p_5.max() - p_5.min())
        # normalize from -0.9 to 0.9
        normolize_phase1 = 1.8 * rescale_phase1 - 0.9
        normolize_phase2 = 1.8 * rescale_phase2 - 0.9
        normolize_phase3 = 1.8 * rescale_phase3 - 0.9
        normolize_phase4 = 1.8 * rescale_phase4 - 0.9
        normolize_phase5 = 1.8 * rescale_phase5 - 0.9
        phase_maps = torch.cat((normolize_phase1, normolize_phase2, normolize_phase3, normolize_phase4, normolize_phase5)).numpy()
        # print(phase_maps.shape)
        # print(dir_name)
        np.save(dir_name[0] + "/" + "phase_maps.npy", phase_maps)
