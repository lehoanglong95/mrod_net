import scipy.io
import numpy as np
from skimage.transform import rotate

def dce_reformat(input_dce, r_thickness, r_distance, r_slices, r_skip_distance, r_image_rotation, number_of_slice, save_dir):
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
    np.save(save_dir, avg_m_output)

if __name__ == '__main__':
    input_dce = scipy.io.loadmat("/home/longlh/hard_2/CMC AI Auto Stroke VOL _Training/BP545 20160701/DMRA5_DCE_Collateral/MatFiles/ColArt.mat")["ColArt"]
    r_thickness = scipy.io.loadmat(
        "/home/longlh/hard_2/CMC AI Auto Stroke VOL _Training/BP545 20160701/DMRA5_DCE_Collateral/MatFiles/dceRTH.mat")[
        "dceRTH"][0][0]
    r_distance = scipy.io.loadmat(
        "/home/longlh/hard_2/CMC AI Auto Stroke VOL _Training/BP545 20160701/DMRA5_DCE_Collateral/MatFiles/dceRDI.mat")[
        "dceRDI"][0][0]
    r_slices = scipy.io.loadmat(
        "/home/longlh/hard_2/CMC AI Auto Stroke VOL _Training/BP545 20160701/DMRA5_DCE_Collateral/MatFiles/dceRSL.mat")[
        "dceRSL"][0][0]
    r_skip_distance = scipy.io.loadmat(
        "/home/longlh/hard_2/CMC AI Auto Stroke VOL _Training/BP545 20160701/DMRA5_DCE_Collateral/MatFiles/dceRSD.mat")[
        "dceRSD"][0][0]
    r_image_rotation = scipy.io.loadmat(
        "/home/longlh/hard_2/CMC AI Auto Stroke VOL _Training/BP545 20160701/DMRA5_DCE_Collateral/MatFiles/dceRIR.mat")[
        "dceRIR"][0][0]
    number_of_slice = scipy.io.loadmat(
        "/home/longlh/hard_2/CMC AI Auto Stroke VOL _Training/BP545 20160701/DMRA5_DCE_Collateral/MatFiles/number_of_slice.mat")[
        "number_of_slice"][0][0]
    dce_reformat(input_dce, r_thickness, r_distance, r_slices, r_skip_distance, r_image_rotation, number_of_slice, "/home/longlh/hard_2/CMC AI Auto Stroke VOL _Training/BP545 20160701/DMRA5_DCE_Collateral/test/avg_m_art.npy")