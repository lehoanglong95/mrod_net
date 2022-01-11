import os
import re
import fnmatch
import pandas as pd

root_path = "/home/longlh/hard_2"
KU = "CMC AI Auto Stroke VOL _Training"
CMC_AI = "Ku AI Auto Stroke VOL _Training"

def generate_valid_folder(output_dir=f"/home/longlh/PycharmProjects/densenet_implementation/valid_folder.csv"):
    ku_dirs = [f"{root_path}/{KU}/{dir}" for dir in os.listdir(f"{root_path}/{KU}")]
    cmc_ai_dirs = [f"{root_path}/{CMC_AI}/{dir}" for dir in os.listdir(f"{root_path}/{CMC_AI}")]
    results = ku_dirs + cmc_ai_dirs
    results = sorted(results, key=lambda x: int(re.search('[0-9]{8}', x).group(0)))
    new_results = []
    for dir in results:
        for file in os.listdir(f"{dir}"):
            if fnmatch.fnmatch(file, '*DCE*'):
                try:
                    checked_files = set(
                        ["IMG.mat", "ColArt.mat", "ColCap.mat", "ColEVen.mat", "ColLVen.mat", "ColDel.mat"])
                    files = set(os.listdir(f"{dir}/{file}/MatFiles"))
                    if checked_files.intersection(files) == checked_files:
                        new_results.append(dir)
                except Exception as e:
                    print(e)
                    continue
    with open(output_dir, 'w') as file:
        for a in new_results:
            file.writelines(f"{a}\n")

def get_datetime_from_filename(x):
    b = re.search('[0-9|-]{8,10}', x).group(0)
    return int(b.replace("-", ""))

def split_dataset(input_file=None, input_base_dir="/home/longlh/hard_2/CMC_KU_AI", output_dir="/home/longlh/PycharmProjects/densenet_implementation/dataset.csv",
                  train_prop=0.5, val_prop=0.2):
    if input_file:
        pd_results = pd.read_csv(input_file, header=None, names=["filenames", "type"])
        normal_results = list(pd_results[pd_results["type"] == "Normal"]["filenames"])
        stroke_results = list(pd_results[pd_results["type"] == "Stroke"]["filenames"])
    else:
        results = [f"{input_base_dir}/{i}" for i in os.listdir(input_base_dir)]
    normal_results = sorted(normal_results, key=get_datetime_from_filename)
    stroke_results = sorted(stroke_results, key=get_datetime_from_filename)
    train_normal_num = int(len(normal_results) * train_prop)
    train_stroke_num = int(len(stroke_results) * train_prop)
    val_normal_num = int(len(normal_results) * val_prop)
    val_stroke_num = int(len(stroke_results) * val_prop)
    train_dataset = normal_results[0:train_normal_num] + stroke_results[0:train_stroke_num]
    val_dataset = normal_results[train_normal_num: train_normal_num + val_normal_num] + stroke_results[train_stroke_num: train_stroke_num + val_stroke_num]
    test_dataset = normal_results[train_normal_num + val_normal_num:] + stroke_results[train_stroke_num + val_stroke_num:]

    # 0: train, 1: val, 2: test
    with open(output_dir, 'w') as file:
        for a in train_dataset:
            file.writelines(f"{a},0\n")
        for b in val_dataset:
            file.writelines(f"{b},1\n")
        for c in test_dataset:
            file.writelines(f"{c},2\n")

if __name__ == '__main__':
    # generate_valid_folder()
    split_dataset(input_file="/home/longlh/PycharmProjects/densenet_implementation/dataset/non_overlap_dataset.csv",
                  input_base_dir=None,
                  output_dir="/home/longlh/PycharmProjects/densenet_implementation/dataset/split_non_overlap_dsc_mrp_dataset.csv")
