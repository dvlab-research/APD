import os
import os.path
import cv2
import numpy as np
import random


from torch.utils.data import Dataset


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(split='train', data_root=None, data_list=None, classes=None):
    assert split in ['train', 'val', 'test']
    if not os.path.isfile(data_list):
        raise (RuntimeError("Image list file do not exist: " + data_list + "\n"))
    image_label_list = []
    list_read = open(data_list).readlines()
    print("Totally {} samples in {} set.".format(len(list_read), split))
    print("Starting Checking image&label pair {} list...".format(split))
    class_analysis = {}
    for cls_idx in range(classes):
        class_analysis[cls_idx] = []
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

        # label = cv2.imread(label_name, cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W
        # unique_label = list(np.unique(label))
        # for tmp_cls in unique_label:
        #     if tmp_cls == 255:
        #         continue
        #     class_analysis[tmp_cls].append(item)
        # if idx % (len(list_read)//20) == 0:
        #     print('Processed {}/{} images...'.format(idx, len(list_read)))

        image_label_list.append(item)
    print("Checking image&label pair {} list done!".format(split))
    return image_label_list, class_analysis

import copy
from PIL import Image
import torchvision
class SemData(Dataset):
    def __init__(self, split='train', data_root=None, data_list=None, transform=None, args=None):
        self.split = split
        self.data_list, self.class_analysis = make_dataset(split, data_root, data_list, classes=args.classes)
        self.transform = transform
        self.args = args

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        image_path, label_path = self.data_list[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        image = np.float32(image)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W

        raw_image = copy.deepcopy(image)
        raw_label = copy.deepcopy(label)
        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Image & label shape mismatch: " + image_path + " " + label_path + "\n"))
        if self.transform is not None:
            image, label = self.transform(image, label)

        cnt = 0
        unique_y = list(label.unique())
        if 255 in unique_y:
            unique_y.remove(255)

        if self.args.low_level_aug:
            aug_image = Image.fromarray(raw_image.astype(np.uint8))
            aug_image = torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)(aug_image)
            raw_image = np.array(aug_image).astype(raw_image.dtype)

        if self.transform is not None:
            new_class_list = []
            while len(set(unique_y).intersection(set(new_class_list))) == 0 and cnt < 50:
                new_chosen_image, new_chosen_label = self.transform(raw_image, raw_label)            
                new_class_list = list(new_chosen_label.unique())
                if 255 in new_class_list:
                    new_class_list.remove(255)
                cnt += 1

        return image, label, new_chosen_image, new_chosen_label
