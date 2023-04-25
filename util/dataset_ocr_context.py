import os
import os.path
import cv2
import numpy as np

from torch.utils.data import Dataset
import torch.nn.functional as F
import random


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(split='train', data_root=None, data_list=None):
    assert split in ['train', 'val', 'test']
    if not os.path.isfile(data_list):
        raise (RuntimeError("Image list file do not exist: " + data_list + "\n"))
    image_label_list = []
    list_read = open(data_list).readlines()
    print("Totally {} samples in {} set.".format(len(list_read), split))
    print("Starting Checking image&label pair {} list...".format(split))
    for line in list_read:
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
        '''
        following check costs some time
        if is_image_file(image_name) and is_image_file(label_name) and os.path.isfile(image_name) and os.path.isfile(label_name):
            item = (image_name, label_name)
            image_label_list.append(item)
        else:
            raise (RuntimeError("Image list file line error : " + line + "\n"))
        '''
        item = (image_name, label_name)
        image_label_list.append(item)
    print("Checking image&label pair {} list done!".format(split))
    return image_label_list


class SemData(Dataset):
    def __init__(self, split='train', data_root=None, data_list=None, transform=None):
        self.split = split
        self.data_list = make_dataset(split, data_root, data_list)
        self.transform = transform
        self.data_root = data_root

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        image_path, label_path = self.data_list[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        image = np.float32(image)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W
        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Image & label shape mismatch: " + image_path + " " + label_path + "\n"))

        unique_small_labels = []
        if self.transform is not None:
            new_image, new_label = self.transform(image, label)            
        ## check input validity
        small_target = F.interpolate(new_label.unsqueeze(0).unsqueeze(0).float(), size=(60, 60), mode='bilinear', align_corners=True).squeeze().long()
        unique_small_labels = list(small_target.unique())
        if 255 in unique_small_labels:
            unique_small_labels.remove(255)
            
        while(len(unique_small_labels) == 0):
            print('reload image, error image path: ', image_path)
            rand_index = random.sample(list(range(len(self.data_list)-1)), 1)[0]
            image_path, label_path = self.data_list[rand_index]
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
            image = np.float32(image)
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W
            if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
                raise (RuntimeError("Image & label shape mismatch: " + image_path + " " + label_path + "\n"))            
            if self.transform is not None:
                new_image, new_label = self.transform(image, label)            
            ## check input validity
            small_target = F.interpolate(new_label.unsqueeze(0).unsqueeze(0).float(), size=(60, 60), mode='bilinear', align_corners=True).squeeze().long()
            unique_small_labels = list(small_target.unique())
            if 255 in unique_small_labels:
                unique_small_labels.remove(255)

        if 'voc2012' in self.data_root:
            ### label_conversion
            my_index = [0, 2, 23, 25, 31, 34, 45, 59, 65, 72, 98, 397, 113, 207, 258, 
                        284, 308, 347, 368, 416, 427, 9, 18, 22, 33, 44, 46, 68, 80, 85, 
                        104, 115, 144, 158, 159, 162, 187, 189, 220, 232, 259, 260, 105, 
                        296, 355, 295, 324, 326, 349, 354, 360, 366, 19, 415, 420, 424, 440, 445, 454, 458]
            hrnet_index = np.sort(np.array([
                        0, 2, 259, 260, 415, 324, 9, 258, 144, 18, 19, 22, 
                        23, 397, 25, 284, 158, 159, 416, 33, 162, 420, 454, 295, 296, 
                        427, 44, 45, 46, 308, 59, 440, 445, 31, 232, 65, 354, 424, 
                        68, 326, 72, 458, 34, 207, 80, 355, 85, 347, 220, 349, 360, 
                        98, 187, 104, 105, 366, 189, 368, 113, 115])).tolist()
            assert len(my_index) == len(hrnet_index)
            raw_label = new_label.clone()
            new_label = new_label.float()
            unique_y = new_label.unique()
            for tmp_cls in unique_y:
                if tmp_cls == 255 or tmp_cls == 0:
                    continue
                src_value = my_index[tmp_cls.long()]
                new_value = hrnet_index.index(src_value)
                tmp_mask = (new_label == tmp_cls).float()
                new_label = new_value * tmp_mask + new_label * (1 - tmp_mask)

        return new_image, new_label.long()
