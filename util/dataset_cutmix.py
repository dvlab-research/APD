import os
import os.path
import cv2
import numpy as np
import time
import random
import torch

from torch.utils.data import Dataset


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
        #self.aux_transform = aux_transform

    def __len__(self):
        return len(self.data_list)

    def rand_bbox(self, size, lam):
        assert len(size) == 4, 'invalid size: {}'.format(size)
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def cutmix_aug(self, image, label, raw_index):
        index = random.randint(0, len(self.data_list)-1)
        while index == raw_index:
            index = random.randint(0, len(self.data_list)-1)
        aux_image_path, aux_label_path = self.data_list[index]
        aux_image = cv2.imread(aux_image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        aux_image = cv2.cvtColor(aux_image, cv2.COLOR_BGR2RGB)  # convert cv2 read aux_image from BGR order to RGB order
        aux_image = np.float32(aux_image)
        aux_label = cv2.imread(aux_label_path, cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W  
        if aux_image.shape[0] != aux_label.shape[0] or aux_image.shape[1] != aux_label.shape[1]:
            raise (RuntimeError("aux_image & aux_label shape mismatch: " + aux_image_path + " " + aux_label_path + "\n"))
        if self.transform is not None:
            aux_image, aux_label = self.transform(aux_image, aux_label)      

        ## generate random cutmix box
        lam = np.random.beta(1.0, 1.0)
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(image.unsqueeze(0).size() if len(image.size()[:]) == 3 else image.size(), lam)
        invalid_mask = (label == 255).float()
        fg_mask = (label != 0).float() * (label != 255).float()
        h, w = label.shape[:]
        inner_cnt = 0
        while (invalid_mask[bbx1:bbx2, bby1:bby2].sum() > 0 or fg_mask[bbx1:bbx2, bby1:bby2].sum() < 8 * 8 * 2 or bbx2 - bbx1 < h*0.25 or bby2 - bby1 < w*0.25) and inner_cnt < 100:
            lam = np.random.beta(1.0, 1.0)
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(image.unsqueeze(0).size() if len(image.size()[:]) == 3 else image.size(), lam)
            inner_cnt += 1
            if inner_cnt >= 100:
                bbx1, bbx2 = 0, int(h * 0.75)
                bby1, bby2 = 0, int(w * 0.75)
                break

        aux_mask = torch.zeros_like(aux_label)
        aux_mask[bbx1:bbx2, bby1:bby2] = 1         
        aux_image[:, bbx1:bbx2, bby1:bby2] = image[:, bbx1:bbx2, bby1:bby2]
        aux_label[bbx1:bbx2, bby1:bby2] = label[bbx1:bbx2, bby1:bby2]

        # ### vis
        # time_str = str(time.time())
        # save_img_path = './vis_dir/{}_img.png'.format(time_str)
        # save_label_path = './vis_dir/{}_label.png'.format(time_str)
        # save_mask_path = './vis_dir/{}_mask.png'.format(time_str)
        # np_image = aux_image.permute(1, 2, 0).numpy()
        # np_label = aux_label.numpy()
        # np_mask = aux_mask.numpy() * 255
        # cv2.imwrite(save_img_path, np_image)
        # cv2.imwrite(save_mask_path, np_mask)
        # print('writing to {}...'.format(save_img_path))

        return aux_image, aux_label, aux_mask
          

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

        aux_image, aux_label, aux_mask = self.cutmix_aug(image, label, raw_index=index)
        #aux_image = self.aux_transform(aux_image)
        return image, label, aux_image, aux_label, aux_mask 
