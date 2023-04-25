import os
import os.path
import cv2
import numpy as np
import random 
import torch

from torch.utils.data import Dataset


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(split='train', data_root=None, data_list=None, classes=21):
    assert split in ['train', 'val', 'test']
    if not os.path.isfile(data_list):
        raise (RuntimeError("Image list file do not exist: " + data_list + "\n"))
    image_label_list = []
    contained_classes = []
    class_files = []
    for i in range(classes):
        class_files.append([])
    list_read = open(data_list).readlines()
    print("Totally {} samples in {} set.".format(len(list_read), split))
    print("Starting Checking image&label pair {} list...".format(split))
    for idx, line in enumerate(list_read):
        line = line.strip()
        line_split = line.split(' ')
        if split == 'test':
            if len(line_split) != 1:
                raise (RuntimeError("Image list file read line error : " + line + "\n"))
            image_name = os.path.join(data_root, line_split[0])
            label_name = image_name  # just set place holder for label_name, not for use
        else:
            if len(line_split) != 2:
                raise (RuntimeError("Image list file read line error : " + line + "\n"))
            image_name = os.path.join(data_root, line_split[0])
            label_name = os.path.join(data_root, line_split[1])

        item = (image_name, label_name)
        image_label_list.append(item)

        ## take the contained classes
        label = cv2.imread(label_name, cv2.IMREAD_GRAYSCALE)
        unique_y = list(np.unique(label))
        for tmp_cls in unique_y:
            if tmp_cls == 255:
                continue
            class_files[tmp_cls].append(idx)
        
        contained_classes.append(unique_y)
    print("Checking image&label pair {} list done!".format(split))
    return image_label_list, class_files


class SemData(Dataset):
    def __init__(self, split='train', data_root=None, data_list=None, transform=None, classes=21):
        self.split = split
        self.data_list, self.class_files = make_dataset(split, data_root, data_list, classes)
        self.transform = transform

    def __len__(self):
        return len(self.data_list) // 2

    def __getitem__(self, index):
        image_path, label_path = self.data_list[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        image = np.float32(image)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W
        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Image & label shape mismatch: " + image_path + " " + label_path + "\n"))
        if self.transform is not None:
            image, label = self.transform(image, label)

        ## get the reference image
        unique_y = list(label.unique())
        if 0 in unique_y:
            unique_y.remove(0)
        if 255 in unique_y:
            unique_y.remove(255)
        if len(unique_y) == 0:
            rand_index = random.randint(0, len(self.data_list)-1)
            ref_image_path, ref_label_path = self.data_list[rand_index]
            ref_image = cv2.imread(ref_image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
            ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)  # convert cv2 read ref_image from BGR order to RGB order
            ref_image = np.float32(ref_image)
            ref_label = cv2.imread(ref_label_path, cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W
            if ref_image.shape[0] != ref_label.shape[0] or ref_image.shape[1] != ref_label.shape[1]:
                raise (RuntimeError("ref_Image & ref_label shape mismatch: " + ref_image_path + " " + ref_label_path + "\n"))
            if self.transform is not None:
                ref_image, ref_label = self.transform(ref_image, ref_label)            
        else:
            class_chosen = unique_y[random.randint(0, len(unique_y)-1)].numpy()
            file_indexes = self.class_files[class_chosen]
            index_chosen = file_indexes[random.randint(0, len(file_indexes)-1)]

            ref_image_path, ref_label_path = self.data_list[index_chosen]   
            raw_ref_label = cv2.imread(ref_label_path, cv2.IMREAD_GRAYSCALE)
            raw_tmp_mask = (raw_ref_label == class_chosen).astype(np.float32)
            assert raw_tmp_mask.sum() > 0

            ref_image = cv2.imread(ref_image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
            ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)  # convert cv2 read ref_image from BGR order to RGB order
            raw_ref_image = np.float32(ref_image)             

            if self.transform is not None:
                ref_image, ref_label = self.transform(raw_ref_image, raw_ref_label)        
                tmp_mask = (ref_label == torch.from_numpy(class_chosen).long()).float()
                while tmp_mask.sum() == 0:
                    ref_image, ref_label = self.transform(raw_ref_image, raw_ref_label)       
                    tmp_mask = (ref_label == torch.from_numpy(class_chosen).long()).float()         

        return image, label, ref_image, ref_label
