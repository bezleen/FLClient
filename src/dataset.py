# --- Open cmt line bellow if run by cmd: python *.py
# import sys  # nopep8
# sys.path.append(".")  # nopep8
# ----
import glob
import torch.utils.data as data
from PIL import Image


class Dataset(data.Dataset):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

    def __init__(self, transform, label_mapping, phase, dataset_path):
        self.transform = transform
        self.phase = phase
        self.label_mapping = label_mapping
        self.file_list = self.datapath_list(dataset_path)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_path = self.file_list[index]
        img = Image.open(img_path)
        img_transformed = self.transform(img, self.TRAIN)
        label = img_path.split(self.phase)[1][1:2]
        num_label = self.label_mapping[label]
        return img_transformed, num_label

    def datapath_list(self, dataset_path):
        # {dataset_path}/train/a/6498eb62131fb0399b0dafe5.jpg
        # {dataset_path}/val/a/6498eb62131fb0399b0dafe5.jpg
        # {dataset_path}/test/a/6498eb62131fb0399b0dafe5.jpg
        target_path = f"{dataset_path}/{self.phase}/**/*.jpg"
        path_list = []
        for path in glob.glob(target_path):
            path_list.append(path)
        return path_list
