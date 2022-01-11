import numpy as np
import hdf5storage
from skimage import exposure
from skimage.filters import threshold_minimum
from skimage.morphology import closing
import csv
import os
from skimage.transform import rotate
import pathlib

def convert_mat_to_npy(mat_file):
    mat_img = hdf5storage.loadmat(mat_file)
    k = mat_file.split("/")[-1].split(".")[0]
    npy_img = mat_img.get(k)
    return npy_img

def transpose_and_normalized(input_imgs):
    output_imgs = np.copy(input_imgs)
    output_imgs = np.moveaxis(output_imgs, 3, 0)
    output_imgs = np.moveaxis(output_imgs, 3, 0)
    output_imgs = (output_imgs - output_imgs.min()) / (output_imgs.max() - output_imgs.min())
    return output_imgs

def generate_mask(input_imgs, percentile=2):
    """
    1. calculate avg intensity based on timeframe axis.
    2. rescale intensity.
    3. use threshold_minimum to generate mask.
    4. use closing morphology operation to improve mask.
    :param input_imgs: ndarray
    :param percentile: int
    :return: mask: ndarray
    """
    _, c, w, h = input_imgs.shape
    input_img = np.sum(input_imgs, axis=0) / input_imgs.shape[0]
    p2, p98 = np.percentile(input_img, (percentile, 100 - percentile))
    input_img = exposure.rescale_intensity(input_img, in_range=(p2, p98))
    masks = []
    for i in range(c):
        thresh = threshold_minimum(input_img[i])
        mask = input_img[i] > thresh
        mask = mask.reshape(1, w, h)
        mask = closing(closing(mask))
        masks.append(mask)
    output_mask = np.concatenate(masks)
    return output_mask.reshape(1, c, w, h)

def preprocess_input(imgs, masks, percentile=2):
    """
    1. rescale intensity in mask with percentile.
    2. rescale to 0 -> 1.
    3. matmul(input, mask).
    :param imgs: ndarray
    :param masks: ndarray
    :param percentile: int
    :return: output_imgs: ndarray
    """
    n, c, w, h = imgs.shape
    outputs = []
    for i in range(n):
        nest_outputs = []
        for j in range(c):
            img = imgs[i][j]
            mask = masks[0][j]
            p2, p98 = np.percentile(img[mask > 0], (percentile, 100 - percentile))
            img = exposure.rescale_intensity(img, in_range=(p2, p98))
            img = img.reshape(1, w, h)
            img = img * mask
            img = (img - img.min()) / (img.max() - img.min())
            nest_outputs.append(img)
        outputs.append(np.concatenate(nest_outputs).reshape(1, c, w, h))
    output_imgs = np.concatenate(outputs)
    return output_imgs

def preprocess_gt(gt, masks, percentile=2):
    """
        1. rescale intensity in mask with percentile.
        2. rescale to -0.9 -> 0.9.
        3. matmul(input, mask).
        :param imgs: ndarray
        :param masks: ndarray
        :param percentile: int
        :return: output_imgs: ndarray
        """
    _, c, w, h = gt.shape
    outputs = []
    for i in range(c):
        img = gt[0][i]
        mask = masks[0][i]
        p2, p98 = np.percentile(img[mask > 0], (percentile, 100 - percentile))
        img = exposure.rescale_intensity(img, in_range=(p2, p98))
        img = img * mask
        img = (img - img.min()) / (img.max() - img.min())
        img = 1.8 * img - 0.9
        # img = signal.medfilt(img, 5).reshape(1, w, h)
        outputs.append(img)
    output_gt = np.concatenate(outputs).reshape(1, c, w, h)
    return output_gt

def take_file_name(dataset_file):
    outputs = []
    with open(dataset_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        for row in csv_reader:
            outputs.append(row[0])
    return outputs

def generate_weight_mask(phase_map_raw, mask, bins=50):
    phase_l = []
    phase_map = np.copy(phase_map_raw)
    for i in range(5):
        phase = phase_map[i]
        real_phase_map = phase[mask[0] > 0]
        temp_weight_map = np.histogram(real_phase_map, bins)
        temp_weight_map[0][temp_weight_map[0] == 0] = 1
        weight_map_tuple = (np.log(1 / (temp_weight_map[0] / real_phase_map.shape[0])), temp_weight_map[1])
        for idx in range(len(weight_map_tuple[0])):
            if idx < len(weight_map_tuple[0]) - 1:
                phase[(phase >= weight_map_tuple[1][idx]) & (phase < weight_map_tuple[1][idx + 1])] = \
                    weight_map_tuple[0][idx]
            else:
                phase[(phase >= weight_map_tuple[1][idx]) & (phase <= weight_map_tuple[1][idx + 1])] = \
                    weight_map_tuple[0][idx]
        d, w, h = phase.shape
        phase = phase.reshape(1, d, w, h)
        phase *= mask[0]
        phase_l.append(phase)
    return np.concatenate(phase_l)

def dce_reformat(r_input_dce, r_thickness, r_distance, r_slices, r_skip_distance, r_image_rotation, number_of_slice):
    input_dce = np.copy(r_input_dce)
    input_shape = input_dce.shape
    height = input_shape[0]
    width = input_shape[1]
    slice = input_shape[3]
    temp_output = np.zeros((height, width, 1, slice))
    for i in range(width):
        temp_output[:, i, 0, :] = rotate(np.squeeze(input_dce[:, i, 0, :]), r_image_rotation)
    m_output = np.zeros((number_of_slice, width, 1, r_slices))
    avg_m_output = np.zeros((number_of_slice, width, 1, r_slices))
    for slice_loop in range(r_slices):
        slice_center = (r_skip_distance + 1) + slice_loop * r_distance
        slice_merge_start = int(slice_center - np.floor(r_thickness / 2))
        slice_merge_end = int(slice_center + (r_thickness - np.floor(r_thickness / 2)) - 1)
        for slice_merge_loop in range(slice_merge_start, slice_merge_end):
            m_output[:, :, 0, slice_loop] = m_output[:, :, 0, slice_loop] + \
                                            np.transpose(np.squeeze(temp_output[slice_merge_loop, :, 0, :]))
        avg_m_output[:, :, 0, slice_loop] = m_output[:, :, 0, slice_loop] / r_thickness
    return avg_m_output


def preprocess(dataset_file, output_base_dir):
    """Preprocess data

    :param dataset_file:
    :param output_base_dir:
    :return: None

    Preprocess pipeline:
    1. convert mat file to npy array.
    2. dce_reformat to down timeframe from ~170 -> 20.
    3. transpose 4d images from front view to top to bottom view then normalize to 0 -> 1.
    4. contrast enhance use rescale intensity then rescale input to 0 -> 1 and ground truth to -0.9 -> 0.9.
    5. save output into output_base_dir.
    """
    file_names = take_file_name(dataset_file)
    total = len(file_names)
    for idx, file_name in enumerate(file_names):
        # print(f"{idx}/{total}")
        try:
            dce_file = [i for i in os.listdir(file_name) if "DCE" in i][0]
            peak_1 = convert_mat_to_npy(os.path.join(file_name, f"{dce_file}", "MatFiles", "dce_peak_phase1.mat"))
            peak_2 = convert_mat_to_npy(os.path.join(file_name, f"{dce_file}", "MatFiles", "dce_peak_phase2.mat"))
            if int(peak_1) >= 25 or int(peak_2) >= 25:
                print(os.path.join(file_name, f"{dce_file}"))
            # print(int(peak_1))
            # print(int(peak_2))
            # patients_idx = file_name.split("/")[-1]
            # dce_file = [i for i in os.listdir(file_name) if "DCE" in i][0]
            # raw_input_imgs = convert_mat_to_npy(os.path.join(file_name, f"{dce_file}", "MatFiles", "IMG.mat"))
            # raw_art_phase = convert_mat_to_npy(os.path.join(file_name, f"{dce_file}", "MatFiles", "ColArt.mat"))
            # raw_cap_phase = convert_mat_to_npy(os.path.join(file_name, f"{dce_file}", "MatFiles", "ColCap.mat"))
            # raw_even_phase = convert_mat_to_npy(os.path.join(file_name, f"{dce_file}", "MatFiles", "ColEVen.mat"))
            # raw_lven_phase = convert_mat_to_npy(os.path.join(file_name, f"{dce_file}", "MatFiles", "ColLVen.mat"))
            # raw_del_phase = convert_mat_to_npy(os.path.join(file_name, f"{dce_file}", "MatFiles", "ColDel.mat"))
            # r_thickness = hdf5storage.loadmat(os.path.join(file_name, f"{dce_file}", "MatFiles", "dceRTH.mat"))["dceRTH"][0][0]
            # r_distance = hdf5storage.loadmat(os.path.join(file_name, f"{dce_file}", "MatFiles", "dceRDI.mat"))[
            #     "dceRDI"][0][0]
            # r_slices = hdf5storage.loadmat(os.path.join(file_name, f"{dce_file}", "MatFiles", "dceRSL.mat"))[
            #     "dceRSL"][0][0]
            # r_skip_distance = hdf5storage.loadmat(os.path.join(file_name, f"{dce_file}", "MatFiles", "dceRSD.mat"))[
            #     "dceRSD"][0][0]
            # try:
            #     r_image_rotation = hdf5storage.loadmat(os.path.join(file_name, f"{dce_file}", "MatFiles", "dceRIR.mat"))[
            #         "dceRIR"][0][0]
            # except Exception as e:
            #     print(e)
            #     r_image_rotation = 0
            # number_of_slice = hdf5storage.loadmat(os.path.join(file_name, f"{dce_file}", "MatFiles", "number_of_slice.mat"))[
            #     "number_of_slice"][0][0]
            # reformated_art_phase = transpose_and_normalized(dce_reformat(raw_art_phase, r_thickness, r_distance, r_slices, r_skip_distance,
            #                                     r_image_rotation, number_of_slice))
            # reformated_cap_phase = transpose_and_normalized(dce_reformat(raw_cap_phase, r_thickness, r_distance, r_slices, r_skip_distance,
            #                                     r_image_rotation, number_of_slice))
            # reformated_even_phase = transpose_and_normalized(dce_reformat(raw_even_phase, r_thickness, r_distance, r_slices, r_skip_distance,
            #                                     r_image_rotation, number_of_slice))
            # reformated_lven_phase = transpose_and_normalized(dce_reformat(raw_lven_phase, r_thickness, r_distance, r_slices, r_skip_distance,
            #                                     r_image_rotation, number_of_slice))
            # reformated_del_phase = transpose_and_normalized(dce_reformat(raw_del_phase, r_thickness, r_distance, r_slices, r_skip_distance,
            #                                     r_image_rotation, number_of_slice))
            # input_imgs_l = []
            # for i in range(raw_input_imgs.shape[2]):
            #     temp_img = raw_input_imgs[:, :, i, :].reshape(raw_input_imgs.shape[0], raw_input_imgs.shape[1], 1, raw_input_imgs.shape[3])
            #     reformated_input_img = dce_reformat(temp_img,
            #                                         r_thickness, r_distance, r_slices, r_skip_distance,
            #                                         r_image_rotation, number_of_slice)
            #     input_imgs_l.append(reformated_input_img)
            # reformated_input_imgs = transpose_and_normalized(np.concatenate(input_imgs_l, axis=2))
            # mask = generate_mask(reformated_input_imgs)
            # preprocessed_input = preprocess_input(reformated_input_imgs, mask)
            # preprocessed_art = preprocess_gt(reformated_art_phase, mask, 1)
            # preprocessed_cap = preprocess_gt(reformated_cap_phase, mask, 2)
            # preprocessed_even = preprocess_gt(reformated_even_phase, mask, 4)
            # preprocessed_lven = preprocess_gt(reformated_lven_phase, mask, 5)
            # preprocessed_del = preprocess_gt(reformated_del_phase, mask, 5)
            # phase_maps = np.concatenate([preprocessed_art, preprocessed_cap, preprocessed_even,
            #                              preprocessed_lven, preprocessed_del])
            # wm = generate_weight_mask(phase_maps, mask)
            # pathlib.Path(os.path.join(output_base_dir, f"{patients_idx}", f"{dce_file}", "MatFiles")).mkdir(parents=True, exist_ok=True)
            # np.save(os.path.join(output_base_dir, f"{patients_idx}", f"{dce_file}", "MatFiles", "mask_4d.npy"), mask)
            # np.save(os.path.join(output_base_dir, f"{patients_idx}", f"{dce_file}", "MatFiles", "IMG_n01.npy"), preprocessed_input)
            # np.save(os.path.join(output_base_dir, f"{patients_idx}", f"{dce_file}", "MatFiles", "phase_maps_rs_intensity_n.npy"), phase_maps)
            # np.save(os.path.join(output_base_dir, f"{patients_idx}", f"{dce_file}", "MatFiles", "phase_maps_rs_intensity_n_wm_50_bins_for_each_phase.npy"), wm)
        except Exception as e:
            print(e)
            print(file_name)
            continue

if __name__ == '__main__':
    preprocess("/home/longlh/PycharmProjects/densenet_implementation/valid_folder.csv", "/home/longlh/hard_2/CMC_KU_AI_New")