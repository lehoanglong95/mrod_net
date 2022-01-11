from torch.utils import data
import csv
import os
import numpy as np
from constants import DatasetType

class NewDscMrpDataset(data.Dataset):

    def __init__(self, csv_file, dataset_type, file_names, transform=None, new_root_dir=None, old_root_dir=None):
        if not isinstance(file_names, dict):
            raise TypeError("file_names must be dict")
        self.new_root_dir = new_root_dir
        self.old_root_dir = old_root_dir
        self.csv_file = csv_file
        self.dataset_type = dataset_type
        self.file_names = file_names
        self.transform = transform
        self._get_file_names(self.csv_file, self.dataset_type, self.file_names)

    def _get_file_names(self, dataset_file, dataset_type, file_names):
        for file_name in file_names.keys():
            setattr(self, file_name, [])
        with open(dataset_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            for row in csv_reader:
                if dataset_type != DatasetType.ALL:
                    if int(row[1]) == dataset_type:
                        if self.old_root_dir and self.new_root_dir:
                            dsc_file = row[0].replace(self.old_root_dir, self.new_root_dir)
                        else:
                            dsc_file = row[0]
                        data_file_dir = dsc_file.replace("/IMG_n01.npy", "")
                        for k, v in file_names.items():
                            getattr(self, k).append(os.path.join(data_file_dir, v))
                else:
                    if self.old_root_dir and self.new_root_dir:
                        dsc_file = row[0].replace(self.old_root_dir, self.new_root_dir)
                    else:
                        dsc_file = row[0]
                    data_file_dir = dsc_file.replace("/IMG_n01.npy", "")
                    for k, v in file_names.items():
                        getattr(self, k).append(os.path.join(data_file_dir, v))

    def __len__(self):
        return len(getattr(self, list(self.file_names.keys())[0]))

    def __getitem__(self, item):
        outputs = []
        for key in self.file_names.keys():
            sample = np.load(getattr(self, key)[item])
            outputs.append(sample)
        if self.transform:
            outputs = self.transform(outputs)
        return tuple(outputs)

    def __repr__(self):
        label_names = [file_name for file_name in self.file_names]
        return "_".join(label_names)

class NewDscMrpDatasetForGenerated(data.Dataset):

    def __init__(self, csv_file, dataset_type, file_names, transform=None, new_root_dir=None, old_root_dir=None):
        if not isinstance(file_names, dict):
            raise TypeError("file_names must be dict")
        self.new_root_dir = new_root_dir
        self.old_root_dir = old_root_dir
        self.csv_file = csv_file
        self.dataset_type = dataset_type
        self.file_names = file_names
        self.transform = transform
        self._get_file_names(self.csv_file, self.dataset_type, self.file_names)

    def _get_file_names(self, dataset_file, dataset_type, file_names):
        self.dirs = []
        for file_name in file_names.keys():
            setattr(self, file_name, [])
        with open(dataset_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            for row in csv_reader:
                if dataset_type != DatasetType.ALL:
                    if int(row[1]) == dataset_type:
                        if self.old_root_dir and self.new_root_dir:
                            dsc_file = row[0].replace(self.old_root_dir, self.new_root_dir)
                        else:
                            dsc_file = row[0]
                        data_file_dir = dsc_file.replace("/IMG_n01.npy", "")
                        self.dirs.append(data_file_dir)
                        for k, v in file_names.items():
                            getattr(self, k).append(os.path.join(data_file_dir, v))
                else:
                    if self.old_root_dir and self.new_root_dir:
                        dsc_file = row[0].replace(self.old_root_dir, self.new_root_dir)
                    else:
                        dsc_file = row[0]
                    data_file_dir = dsc_file.replace("/IMG_n01.npy", "")
                    self.dirs.append(data_file_dir)
                    for k, v in file_names.items():
                        getattr(self, k).append(os.path.join(data_file_dir, v))

    def __len__(self):
        return len(getattr(self, list(self.file_names.keys())[0]))

    def __getitem__(self, item):
        outputs = []
        for key in self.file_names.keys():
            sample = np.load(getattr(self, key)[item])
            outputs.append(sample)
        if self.transform:
            outputs = self.transform(outputs)
        outputs.append(self.dirs[item])
        return tuple(outputs)

    def __repr__(self):
        label_names = [file_name for file_name in self.file_names]
        return "_".join(label_names)

if __name__ == '__main__':
    from constants import DatasetType
    a = NewDscMrpDataset("/home/longlh/PycharmProjects/densenet_implementation/dataset/split_dsc_mrp_dataset.csv",
                         DatasetType.TRAIN, {"a": "IMG_n01.npy"})
    for b in a:
        print(b[0].shape)