from torch.utils import data
import numpy as np
import os
from dsc_mrp_dataset_for_generate import DatasetType, DscMrpDataset
import scipy.ndimage as ndi
from multiprocessing import Process


def median_filter(x):
    a = x[0].numpy()[0]
    dir = x[1]
    channels, depth, width, height = a.shape
    for c in range(channels):
        for d in range(depth):
            a[c,d] = ndi.median_filter(a[c, d], 5)
    np.save(f'{dir[0]}/IMG_n01_medfilt.npy', a)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    params = {"batch_size": 1,
              "shuffle": False,
              "num_workers": 0}
    # '/media/data1/longlh', '/home/quiiubuntu/longlh/patient_list.xlsx'
    # '/home/longlh/Desktop', '/home/longlh/Downloads/patient_list.xlsx'
    training_dataset = DscMrpDataset('/home/longlh/Desktop', '/home/longlh/Downloads/patient_list.xlsx',
                                     DatasetType.TEST, {'inputs': 'IMG_n01.npy'}, None)
    training_generator = data.DataLoader(training_dataset, **params)
    # inputs = defaultdict(list)
    training_generator_iter = iter(training_generator)
    print(len(training_generator))
    def generator():
        for idx in range(0, len(training_generator), 4):
            yield (next(training_generator_iter),
                   next(training_generator_iter))
    # for idx, (input, dirs) in enumerate(training_generator):
    #     key = idx // 8
    #     inputs[key].append((input[0].numpy(), dirs))
    for inputs in generator():
        inputs_l = list(inputs)
        procs = []
        for input in inputs_l:
            proc = Process(target=median_filter, args=(input,))
            procs.append(proc)
            proc.start()
        for proc in procs:
            proc.join()
        # inputs = inputs.numpy()
        # for input in inputs:
        #     proc = Process(target=median_filter, args=(input, ))
        #     procs.append(proc)
        #     proc.start()
    # for proc in procs:
    #     proc.join()