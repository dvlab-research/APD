import torch
from torch import nn
import torch.nn.functional as F

import model.resnet as models

torch.manual_seed(123)
torch.cuda.manual_seed(123)
torch.cuda.manual_seed_all(123)

class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)
        self.merge = nn.Sequential(
            nn.Conv2d(in_dim * 2, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
        )
    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return self.merge(torch.cat(out, 1))


class Model(nn.Module):
    def __init__(self, layers=50, bins=(1, 2, 3, 6), dropout=0.1, classes=2, zoom_factor=8, use_ppm=True, \
                criterion=nn.CrossEntropyLoss(ignore_index=255), pretrained=True, args=None):
        super().__init__()
        assert layers in [18, 50, 101, 152]
        assert 2048 % len(bins) == 0
        assert classes > 1
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor
        self.use_ppm = use_ppm
        self.criterion = criterion
        self.args = args

        if layers == 18:
            resnet = models.resnet18(pretrained=pretrained)
        elif layers == 50:
            resnet = models.resnet50(pretrained=pretrained)
        elif layers == 101:
            resnet = models.resnet101(pretrained=pretrained)
        else:
            resnet = models.resnet152(pretrained=pretrained)
        self.relu = nn.ReLU()
        if layers < 50:
            self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.conv2, resnet.bn2, resnet.relu, resnet.conv3, resnet.bn3, resnet.relu, resnet.maxpool)
            key_conv = 'conv1'
            fea_dim = 512
            aux_dim = 256
        else:
            self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.conv2, resnet.bn2, resnet.relu, resnet.conv3, resnet.bn3, resnet.relu, resnet.maxpool)
            key_conv = 'conv2'
            fea_dim = 2048
            aux_dim = 1024
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        if args.classes == 60:
            for n, m in self.layer3.named_modules():
                if 'conv' in n:
                    if m.stride == (2, 2):
                        m.stride = (1, 1)
                        if m.kernel_size == (3, 3):
                            m.dilation = (1, 1)
                            m.padding = (1, 1)                    
                    else:
                        if m.kernel_size == (3, 3):
                            m.dilation = (2, 2)
                            m.padding = (2, 2)   
                elif  'downsample' in n:
                    m.stride = (1, 1)                                         

            for n, m in self.layer4.named_modules():
                if 'conv' in n:
                    if m.stride == (2, 2):
                        m.stride = (1, 1)
                        if m.kernel_size == (3, 3):
                            m.dilation = (2, 2)
                            m.padding = (2, 2)                    
                    else:
                        if m.kernel_size == (3, 3):
                            m.dilation = (4, 4)
                            m.padding = (4, 4)         
                elif  'downsample' in n:
                    m.stride = (1, 1)          
        else:  
            for n, m in self.layer3.named_modules():
                if key_conv in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
            for n, m in self.layer4.named_modules():
                if key_conv in n:
                    m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)

        if use_ppm:
            self.ppm = PPM(fea_dim, int(fea_dim/len(bins)), bins)
            fea_dim *= 2
        self.cls = nn.Sequential(
            nn.Conv2d(512, classes, kernel_size=1)
        )
        if self.training:
            self.aux = nn.Sequential(
                nn.Conv2d(aux_dim, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(256, classes, kernel_size=1)
            )
        self.layers = layers   

    def forward(self, x, y=None, scale_aug=None):
        x_size = x.size()
        assert (x_size[2]-1) % 8 == 0 and (x_size[3]-1) % 8 == 0
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        x = self.layer0(x)
        x0 = x.clone()
        x = self.layer1(x)
        x1 = x.clone()
        x = self.layer2(x)
        x2 = x.clone()
        x_tmp = self.layer3(x)
        x3 = x_tmp.clone()
        x = self.layer4(x_tmp)
        x4 = x.clone()
        if self.use_ppm:
            x = self.ppm(x)
        x_ppm = x.clone()
        x = self.cls(x)
        x_final = x.clone()
        if self.zoom_factor != 1: 
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

        if y is not None:
            aux = self.aux(x_tmp)
            if self.zoom_factor != 1:
                aux = F.interpolate(aux, size=(h, w), mode='bilinear', align_corners=True)

            if scale_aug:
                aug_num = x.shape[0] // 2
                main_loss = self.criterion(x[:aug_num], y[:aug_num])
                aux_loss = self.criterion(aux[:aug_num], y[:aug_num])                 
            else:
                main_loss = self.criterion(x, y)
                aux_loss = self.criterion(aux, y)               

            return x.max(1)[1], main_loss, aux_loss, [x0, x1, x2, x3, x4, x_ppm, x_final]
        else:
            return x

