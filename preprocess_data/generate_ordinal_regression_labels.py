
import torch
from torch.utils import data
import numpy as np
import os
# from dsc_mrp_dataset_for_generate import DscMrpDataset
from constants import DatasetType
import copy

# uniform discretization
def ud(K, raw_depth):
    depth_tensor = copy.deepcopy(raw_depth)
    depth_tensor = (depth_tensor - depth_tensor.min()) / (depth_tensor.max() - depth_tensor.min())
    alpha = torch.tensor(depth_tensor.min())
    beta = torch.tensor(depth_tensor.max())
    threshold_l = []
    for i in range(K + 1):
        threshold_l.append(alpha + (beta - alpha) * i / K)
    masks = []
    for i in range(len(threshold_l) - 1):
        if i == len(threshold_l) - 2:
            masks.append((depth_tensor >= threshold_l[i]))
        else:
            masks.append(((depth_tensor >= threshold_l[i]) & (depth_tensor < threshold_l[i+1])))
    for i in range(len(masks)):
        depth_tensor[masks[i]] = i
    return depth_tensor.type(torch.IntTensor)
    # c = (depth_tensor - alpha) * K / (beta - alpha)
    # return c.type(torch.IntTensor)

def sd(alpha, beta, K, depth_tensor):
    c = K * torch.log(depth_tensor / torch.tensor(alpha)) / torch.log(torch.tensor(beta) / torch.tensor(alpha))
    return c.type(torch.IntTensor)


def imbalance_discretization(alpha, beta, K, proportions, depth):
    # add beta to avoid conflict after iteration
    depth_copy = torch.clone(depth).detach()
    for idx, proportion in enumerate(proportions):
        depth_copy[depth_copy < torch.tensor(alpha) + (torch.tensor(beta) - torch.tensor(alpha)) * torch.tensor(proportion)] = idx + beta + 1
    depth_copy[depth_copy == beta] = K - 1 + beta + 1
    max_thres = torch.tensor(alpha) + (torch.tensor(beta) - torch.tensor(alpha)) * torch.tensor(proportions[len(proportions) - 1])
    depth_copy[torch.mul(max_thres <= depth_copy, depth_copy < beta)] = K - 2 + beta + 1
    # remove beta from depth
    return depth_copy - beta - 1


def spacing_increasing_discretization(K, depth):
    new_depth = (depth - depth.min()) / (depth.max() - depth.min())
    new_depth = torch.clamp(new_depth, min=0.001, max=1)
    alpha = torch.tensor(0.001)
    beta = torch.tensor(new_depth.max())
    K = torch.tensor(K)

    if torch.cuda.is_available():
        alpha = alpha.cuda()
        beta = beta.cuda()
        K = K.cuda()
    labels = K * torch.log(new_depth / alpha) / torch.log(beta / alpha)
    if torch.cuda.is_available():
        labels = labels.cuda()
    return labels.int()

def convert_sid_to_depth(K, labels):
    min = 0.001
    max = 1

    if torch.cuda.is_available():
        alpha_ = torch.tensor(min).cuda()
        beta_ = torch.tensor(max).cuda()
        K_ = torch.tensor(K).cuda()
    else:
        alpha_ = torch.tensor(min)
        beta_ = torch.tensor(max)
        K_ = torch.tensor(K)

    old_min = min
    old_max = max
    new_min = -0.9
    new_max = 0.9
    depth = torch.exp(torch.log(alpha_) + (torch.log(beta_ / alpha_) * labels / K))
    depth_range = torch.unique(depth)
    for i in range(len(depth_range) -1):
        depth[(depth >= depth_range[i]) & (depth < depth_range[i + 1])] = (depth_range[i] + depth_range[i + 1]) / 2
    depth = ((depth - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
    # print(depth.size())
    return depth.float()

def spacing_decreasing_discretization(K, raw_depth):
    depth = copy.deepcopy(raw_depth)
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    alpha = torch.tensor(depth.min())
    beta = torch.tensor(depth.max())
    K = torch.tensor(K)
    if torch.cuda.is_available():
        alpha = alpha.cuda()
        beta = beta.cuda()
        K = K.cuda()
    labels = (depth ** 2 - alpha ** 2) * K / (beta ** 2 - alpha ** 2)
    if torch.cuda.is_available():
        labels = labels.cuda()
    return labels.int()

"""
    split range into 2 equal subranges
    
"""

def choose_threshold(alpha, beta, K):
   # 3 range -> 4 pivot threshold
   new_alpha = 0.001 if alpha == 0 else alpha
   threshold_list = []
   for k in range(K + 1):
        if k == 0:
            threshold_list.append(alpha)
        elif k == K:
            threshold_list.append(beta)
        else:
            temp_new_alpha = torch.tensor(new_alpha)
            temp_beta = torch.tensor(beta)
            thres = torch.exp(torch.log(temp_new_alpha) + (torch.log(temp_beta / temp_new_alpha) * k / K))
            threshold_list.append(float(thres))
   return threshold_list

def double_sid(K, raw_depth):
    """
    1. split range into 2 equal subranges.
    2. calculate threshold for first subranges.
    3. calculate distance from first threshold list to alpha -> calculate threshold for second subranges.
    4. calculate K labels.
    For examples with K = 5
    t0-t1--t2------t3--t4-t5
    :param K: number of classes
    :param raw_depth: depth_tensor
    :return: K labels tensor
    """
    depth_tensor = copy.deepcopy(raw_depth)
    depth_tensor = (depth_tensor - depth_tensor.min()) / (depth_tensor.max() - depth_tensor.min())
    alpha = float(depth_tensor.min())
    beta = float(depth_tensor.max())
    middle = (alpha + beta) / 2
    first_list_thresholds = choose_threshold(alpha, middle, int(K // 2 + 1))
    distance_from_first_list_thresholds_to_alpha = [x - alpha for x in first_list_thresholds]
    invert_second_list_thresholds = [beta - x for x in distance_from_first_list_thresholds_to_alpha]
    resorted_invert_second_list_threshold = invert_second_list_thresholds[::-1]
    final_threshold = list(set(first_list_thresholds).symmetric_difference(resorted_invert_second_list_threshold))
    final_threshold.sort()
    masks = []
    for i in range(len(final_threshold) - 1):
        if i == len(final_threshold) - 2:
            masks.append((depth_tensor >= final_threshold[i]))
        else:
            masks.append(((depth_tensor >= final_threshold[i]) & (depth_tensor < final_threshold[i + 1])))
    for i in range(len(masks)):
        depth_tensor[masks[i]] = i
    return depth_tensor.type(torch.IntTensor)

def polynomial_symmetric_discretization(K, depth):
    new_depth = (depth - depth.min()) / (depth.max() - depth.min())
    alpha = 0
    beta = 0.5
    K = K // 2
    threshold_arr = []
    for i in range(K+1):
        threshold_arr.append(round((np.sqrt(alpha) + (np.sqrt(beta) - np.sqrt(alpha)) * i / K) ** 2, 3))
        threshold_arr.append(round(1 - (np.sqrt(alpha) + (np.sqrt(beta) - np.sqrt(alpha)) * (K-i) / K) ** 2, 3))
    threshold_arr.sort()
    threshold_s = list(set(threshold_arr))
    threshold_s.sort()
    output_depth = np.zeros(new_depth.shape)
    for i in range(len(threshold_s)-1):
        if i == len(threshold_s) - 2:
            output_depth[new_depth >= threshold_s[i]] = i
        else:
            output_depth[(new_depth >= threshold_s[i]) & (new_depth < threshold_s[i+1])] = i
    return output_depth.astype(int)

def square_root_symmetric_discretization(K, depth):
    new_depth = (depth- depth.min()) / (depth.max() - depth.min())
    alpha = 0
    middle = 0.5
    beta = 1
    K = K // 2
    threshold_arr = []
    for i in range(K + 1):
        threshold_arr.append(round(middle - np.sqrt(alpha ** 2 + (middle ** 2 - alpha ** 2) * i / K), 3))
        threshold_arr.append(round(np.sqrt(middle ** 2 + (beta ** 2 - middle ** 2) * i / K), 3))
    threshold_arr.sort()
    threshold_s = list(set(threshold_arr))
    threshold_s.sort()
    output_depth = np.zeros(new_depth.shape)
    for i in range(len(threshold_s) - 1):
        if i == len(threshold_s) - 2:
            output_depth[new_depth >= threshold_s[i]] = i
        else:
            output_depth[(new_depth >= threshold_s[i]) & (new_depth < threshold_s[i + 1])] = i
    return output_depth.astype(int)

if __name__ == '__main__':
    # import numpy as np
    # a = np.load("/home/longlh/hard_2/Workspace/KU AIS Patient Anonymized/IAT AIS_pros/BP10 20170122 Lt multi M2 occ/20170122 PRE-IAT 02051182/AX_PWI5_DSC_Collateral/phase_maps_medfilt_rs_n.npy")
    # a = torch.from_numpy(a)
    # b = spacing_increasing_discretization(199, a)
    # c = convert_sid_to_depth(199, b)
    # c = c.to("cpu")
    # print(c)
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    params = {"batch_size": 1,
              "shuffle": False,
              "num_workers": 0}
    # from dce_mrp_dataset import DceMrpDatasetForGenerated
    from dsc_mrp_dataset_for_generate import DscMrpDataset
    from new_dsc_mrp_dataset import NewDscMrpDatasetForGenerated
    # '/media/data1/longlh', '/home/quiiubuntu/longlh/patient_list.xlsx'
    # '/home/longlh/Desktop', '/home/longlh/Downloads/patient_list.xlsx'
    # training_dataset = NewDscMrpDatasetForGenerated("/home/longlh/PycharmProjects/densenet_implementation/dataset/split_non_overlap_dsc_mrp_dataset.csv",
    #                      DatasetType.ALL, {'labels': 'phase_maps_medfilt_rs_n.npy'}, old_root_dir="/home/longlh/hard_2/Workspace",
    #                                                 new_root_dir="/home/longlh/hard_2/Workspace")
    training_dataset = NewDscMrpDatasetForGenerated(
        "/data1/long/longlh/longlh/densenet_implementation/dataset/split_non_overlap_dsc_mrp_dataset.csv",
        DatasetType.ALL, {'labels': 'phase_maps_medfilt_rs_n.npy'}, old_root_dir="/home/longlh/hard_2/Workspace",
        new_root_dir="/data1/long/longlh/longlh/Workspace")
    # training_dataset = DceMrpDatasetForGenerated("/home/longlh/hard_2", "dataset.csv", 3,
    #                                  {'labels': 'phase_maps_rs_intensity_n.npy'}, None)
    training_generator = data.DataLoader(training_dataset, **params)
    #
    for (label, dirs) in training_generator:
        #label = label.cuda()
        label = label.numpy()
        # label_ud = polynomial_symmetric_discretization(10, label)
        label_ordinal = square_root_symmetric_discretization(5, label)
        #label_ordinal = spacing_decreasing_discretization(4, label)
        np.save(f'{dirs[0]}/phase_maps_or_square_root_symmetric_discretization_label_medfilt_rs_5classes_n.npy',
                label_ordinal[0])

    #TEST
    # a = torch.Tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    # b = ud(0, 1, 5, a)
    # print(b)
