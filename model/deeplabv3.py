import torch
from torch import nn
import torch.nn.functional as F

import model.resnet as models

torch.manual_seed(123)
torch.cuda.manual_seed(123)
torch.cuda.manual_seed_all(123)

from mmcv.cnn import ConvModule
#from mmseg.ops import resize


class ASPPModule(nn.ModuleList):
    """Atrous Spatial Pyramid Pooling (ASPP) Module.

    Args:
        dilations (tuple[int]): Dilation rate of each layer.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
    """

    def __init__(self, dilations, in_channels, channels, conv_cfg, norm_cfg,
                 act_cfg):
        super(ASPPModule, self).__init__()
        self.dilations = dilations
        self.in_channels = in_channels
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        for dilation in dilations:
            self.append(
                ConvModule(
                    self.in_channels,
                    self.channels,
                    1 if dilation == 1 else 3,
                    dilation=dilation,
                    padding=0 if dilation == 1 else dilation,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

    def forward(self, x):
        """Forward function."""
        aspp_outs = []
        for aspp_module in self:
            aspp_outs.append(aspp_module(x))

        return aspp_outs

#class ASPPHead(BaseDecodeHead):
class ASPPHead(nn.Module):
    """Rethinking Atrous Convolution for Semantic Image Segmentation.

    This head is the implementation of `DeepLabV3
    <https://arxiv.org/abs/1706.05587>`_.

    Args:
        dilations (tuple[int]): Dilation rates for ASPP module.
            Default: (1, 6, 12, 18).
    """

    def __init__(self, dilations,                 
                 num_classes,
                 in_channels,
                 channels,
                 dropout_ratio=0.1,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 in_index=-1,
                 input_transform=None,
                 ignore_index=255,
                 sampler=None,
                 align_corners=False):
        super(ASPPHead, self).__init__()
        #super(ASPPHead, self).__init__(**kwargs)
        assert isinstance(dilations, (list, tuple))
        self.in_channels = in_channels
        self.dilations = dilations
        self.channels = channels
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.in_index = in_index
        self.ignore_index = ignore_index
        self.align_corners = align_corners

        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvModule(
                self.in_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        self.aspp_modules = ASPPModule(
            dilations,
            self.in_channels,
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.bottleneck = ConvModule(
            (len(dilations) + 1) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None


    def forward(self, inputs):
        """Forward function."""
        x = inputs
        aspp_outs = [
            F.interpolate(
                self.image_pool(x),
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        ]
        aspp_outs.extend(self.aspp_modules(x))
        aspp_outs = torch.cat(aspp_outs, dim=1)
        output = self.bottleneck(aspp_outs)
        if self.dropout is not None:
            output = self.dropout(output)
        return output


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates, BatchNorm):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), BatchNorm(out_channels), nn.ReLU()))
        for atrous_rate in atrous_rates:
            modules.append(nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate, dilation=atrous_rate, bias=False), BatchNorm(out_channels), nn.ReLU()))
        modules.append(nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels, out_channels, 1, bias=False), BatchNorm(out_channels), nn.ReLU()))
        self.convs = nn.ModuleList(modules)

        self.merge = nn.Sequential(
            nn.Conv2d(out_channels * (len(atrous_rates)+2), 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
        )
    def forward(self, x):
        x_size = x.size()        
        res = []
        for conv in self.convs[:-1]:
            res.append(conv(x))
        res.append(F.interpolate(self.convs[-1](x), x_size[2:], mode='bilinear', align_corners=True))
        return self.merge(torch.cat(res, dim=1))


class PSPNet(nn.Module):
    def __init__(self, layers=50, bins=(1, 2, 3, 6), dropout=0.1, classes=2, zoom_factor=8, use_ppm=True, \
                criterion=nn.CrossEntropyLoss(ignore_index=255), pretrained=True, args=None):
        super(PSPNet, self).__init__()
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
            #atrous_rates = (6, 12, 18)
            #atrous_rates = (12, 24, 36)
            #self.ppm = ASPP(fea_dim, int(fea_dim/(len(atrous_rates)+1)), atrous_rates, BatchNorm=nn.BatchNorm2d)
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


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
    input = torch.rand(4, 3, 473, 473).cuda()
    model = PSPNet(layers=50, bins=(1, 2, 3, 6), dropout=0.1, classes=21, zoom_factor=1, use_ppm=True, pretrained=True).cuda()
    model.eval()
    print(model)
    output = model(input)
    print('PSPNet', output.size())
