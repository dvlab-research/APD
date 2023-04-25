import torch
from torch import nn
import torch.nn.functional as F

#import model.resnet as models
from .effnet import EfficientNet
from .deeplabv3 import ASPPHead 

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
                criterion=nn.CrossEntropyLoss(ignore_index=255), pretrained=True, args=None, use_aspp=False):
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
            assert False
        elif layers == 50:
            self.effnet = EfficientNet.from_pretrained('efficientnet-b0', weights_path = 'initmodel/efficientnet-b0-355c32eb.pth')
        else: 
            assert False
        if layers < 50:
            key_conv = 'conv1'
            fea_dim = 160
            aux_dim = 48
        else:
            key_conv = 'conv2'
            fea_dim = 1280
            aux_dim = 192

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
        '''
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
        '''
        if use_ppm:
            if use_aspp:
                self.ppm = ASPPHead(
                    dilations=(1, 12, 24, 36),
                    #dilations=(1, 6, 12, 18),
                    in_channels=fea_dim,
                    in_index=3,
                    channels=512,
                    dropout_ratio=0.1,
                    num_classes=19,
                    align_corners=True,
                    norm_cfg = dict(type='BN', requires_grad=True)
                )
            else:
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

        feat = self.effnet(x)
        if self.use_ppm:
            x = self.ppm(feat[4])
        x_ppm = x.clone()
        x = self.cls(x)
        x_final = x.clone()
        if self.zoom_factor != 1:
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

        if y is not None:
            aux = self.aux(feat[3])
            if self.zoom_factor != 1:
                aux = F.interpolate(aux, size=(h, w), mode='bilinear', align_corners=True)

            if scale_aug:
                aug_num = x.shape[0] // 2
                main_loss = self.criterion(x[:aug_num], y[:aug_num])
                aux_loss = self.criterion(aux[:aug_num], y[:aug_num])                 
            else:
                main_loss = self.criterion(x, y)
                aux_loss = self.criterion(aux, y)               

            return x.max(1)[1], main_loss, aux_loss, [feat[0], feat[1], feat[2], feat[3], feat[4], x_ppm, x_final]
        else:
            return x


