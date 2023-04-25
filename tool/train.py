import os
import random
import time
import cv2
import numpy as np
import logging
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
from tensorboardX import SummaryWriter

from util import dataset_fix as dataset, transform, config
from util.util import AverageMeter, poly_learning_rate, intersectionAndUnionGPU, find_free_port

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/ade20k/ade20k_pspnet50.yaml', help='config file')
    parser.add_argument('opts', help='see config/ade20k/ade20k_pspnet50.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def main_process():
    return not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)


def check(args):
    assert args.classes > 1
    assert args.zoom_factor in [1, 2, 4, 8]
    assert args.arch in ['psp', 'pspmv2', 'pspeffnet', 'deeplabv3', 'deeplabv3mv2', 'deeplabv3effnet']
    assert (args.train_h - 1) % 8 == 0 and (args.train_w - 1) % 8 == 0

def main():
    args = get_parser()
    check(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)
    if len(args.train_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False
    if args.multiprocessing_distributed:
        port = find_free_port()
        args.dist_url = f"tcp://127.0.0.1:{port}"
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        main_worker(args.train_gpu, args.ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, argss):
    global args
    args = argss
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)
    if args.stu_trans:
        StuTransConv = nn.Sequential(
            nn.Conv2d(512, args.kmeans_adapt_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(args.kmeans_adapt_dim, 512, kernel_size=1),
        )
    else:
        StuTransConv = nn.Identity()
    if args.tea_trans:
        TeaTransConv = nn.Sequential(
            nn.Conv2d(512, args.kmeans_adapt_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(args.kmeans_adapt_dim, 512, kernel_size=1),
        )                    
    else:
        TeaTransConv = nn.Identity()
    teacher_modules_list = [TeaTransConv]
    if args.arch == 'psp':
        from model.pspnet import Model as Model_s
        model = Model_s(layers=args.layers, classes=args.classes, zoom_factor=args.zoom_factor, criterion=criterion, args=args)
        modules_ori = [model.layer0, model.layer1, model.layer2, model.layer3, model.layer4]
        modules_new = [model.ppm, model.cls, model.aux, StuTransConv]
    elif args.arch == 'pspmv2':
        from model.pspnet_mv2 import Model as Model_s
        model = Model_s(widen_factor=args.widen_factor, classes=args.classes, zoom_factor=args.zoom_factor, criterion=criterion, args=args)
        modules_ori = [model.mv2]
        modules_new = [model.ppm, model.cls, model.aux, StuTransConv]
    elif args.arch == 'pspeffnet':
        from model.pspnet_effnet import Model as Model_s
        model = Model_s(layers=args.layers, classes=args.classes, zoom_factor=args.zoom_factor, criterion=criterion, args=args)
        modules_ori = [model.effnet]
        modules_new = [model.ppm, model.cls, model.aux, StuTransConv]
    elif args.arch == 'deeplabv3':
        from model.deeplabv3 import Model as Model_s
        model = Model_s(layers=args.layers, classes=args.classes, zoom_factor=args.zoom_factor, criterion=criterion, args=args)
        modules_ori = [model.layer0, model.layer1, model.layer2, model.layer3, model.layer4]
        modules_new = [model.ppm, model.cls, model.aux, StuTransConv]
    elif args.arch == 'deeplabv3mv2':
        from model.pspnet_fc_mv2 import Model as Model_s
        model = Model_s(layers=args.layers, classes=args.classes, zoom_factor=args.zoom_factor, criterion=criterion, args=args, use_aspp=True)
        modules_ori = [model.mv2]
        modules_new = [model.ppm, model.cls, model.aux, StuTransConv]
    elif args.arch == 'deeplabv3effnet':
        from model.pspnet_fc_effnet import Model as Model_s
        model = Model_s(layers=args.layers, classes=args.classes, zoom_factor=args.zoom_factor, criterion=criterion, args=args, use_aspp=True)
        modules_ori = [model.effnet]
        modules_new = [model.ppm, model.cls, model.aux, StuTransConv]
    if args.arch_t == 'psp':
        from model.pspnet import Model as Model_t
        t_model = Model_t(layers=args.t_layers, classes=args.classes, zoom_factor=args.zoom_factor, criterion=criterion, args=args)
    elif args.arch_t == 'pspmv2':
        from model.pspnet_mv2 import Model as Model_t
        t_model = Model_t(layers=args.t_layers, classes=args.classes, zoom_factor=args.zoom_factor, criterion=criterion, args=args)
    elif args.arch_t == 'deeplabv3':
        from model.deeplabv3 import Model as Model_t
        t_model = Model_t(layers=args.t_layers, classes=args.classes, zoom_factor=args.zoom_factor, criterion=criterion, args=args)
    elif args.arch_t == '':
        t_model = None

    if t_model is not None:
        t_model.eval()

    params_list = []
    for module in modules_ori:
        params_list.append(dict(params=module.parameters(), lr=args.base_lr))
    for module in modules_new:
        params_list.append(dict(params=module.parameters(), lr=args.base_lr * 10))
    args.index_split = 5
    optimizer = torch.optim.SGD(params_list, lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)

    teacher_param_list = []
    for module in teacher_modules_list:
        teacher_param_list.append(dict(params=module.parameters(), lr=args.teacher_lr))
    teacher_optimizer = torch.optim.Adam(teacher_param_list, args.teacher_lr, [0.9, 0.99])

    if args.sync_bn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        StuTransConv = nn.SyncBatchNorm.convert_sync_batchnorm(StuTransConv)
        TeaTransConv = nn.SyncBatchNorm.convert_sync_batchnorm(TeaTransConv)


    if main_process():
        global logger, writer
        logger = get_logger()
        writer = SummaryWriter(args.save_path)
        logger.info(args)
        logger.info("=> creating Teacher model ...")
        logger.info("Classes: {}".format(args.classes))
        logger.info(t_model)        
        logger.info("=> creating Student model ...")
        logger.info("Classes: {}".format(args.classes))
        logger.info(model)
        logger.info("=> creating TeaTrans model ...")
        logger.info(TeaTransConv)
    if args.distributed:
        torch.cuda.set_device(gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.batch_size_val = int(args.batch_size_val / ngpus_per_node)
        args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[gpu])
        if t_model is not None:
            t_model = torch.nn.parallel.DistributedDataParallel(t_model.cuda(), device_ids=[gpu])
        if args.stu_trans:
            StuTransConv = torch.nn.parallel.DistributedDataParallel(StuTransConv.cuda(), device_ids=[gpu])
        if args.tea_trans:
            TeaTransConv = torch.nn.parallel.DistributedDataParallel(TeaTransConv.cuda(), device_ids=[gpu])
    else:
        model = torch.nn.DataParallel(model.cuda())
        if t_model is not None:
            t_model = torch.nn.DataParallel(t_model.cuda())
        StuTransConv = torch.nn.DataParallel(StuTransConv.cuda())
        TeaTransConv = torch.nn.DataParallel(TeaTransConv.cuda())

    if args.weight:
        if os.path.isfile(args.weight):
            if main_process():
                logger.info("=> loading weight '{}'".format(args.weight))
            #checkpoint = torch.load(args.weight)
            if len(args.train_gpu) > 1:
                map_location = {'cuda:0': 'cuda:{}'.format(args.rank)} if args.rank > 0 else None
                checkpoint = torch.load(args.weight, map_location=map_location)
            else:
                checkpoint = torch.load(args.weight)
            model.load_state_dict(checkpoint['state_dict'])
            if main_process():
                logger.info("=> loaded weight '{}'".format(args.weight))
        else:
            if main_process():
                logger.info("=> no weight found at '{}'".format(args.weight))

    if main_process():
        logger.info('===> initializing teacher model from : {}'.format(args.t_weights))

    if args.t_weights:
        print(t_model.load_state_dict(torch.load(args.t_weights)['state_dict'], strict=False))

    if args.resume:
        if os.path.isfile(args.resume):
            if main_process():
                logger.info("=> loading checkpoint '{}'".format(args.resume))
            # checkpoint = torch.load(args.resume)
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if main_process():
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            if main_process():
                logger.info("=> no checkpoint found at '{}'".format(args.resume))

    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    train_transform = transform.Compose([
        transform.RandScale([args.scale_min, args.scale_max]),
        transform.RandRotate([args.rotate_min, args.rotate_max], padding=mean, ignore_label=args.ignore_label),
        transform.RandomGaussianBlur(),
        transform.RandomHorizontalFlip(),
        transform.Crop([args.train_h, args.train_w], crop_type='rand', padding=mean, ignore_label=args.ignore_label),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)])
    train_data = dataset.SemData(split='train', data_root=args.data_root, data_list=args.train_list, transform=train_transform)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    if args.evaluate:
        val_transform = transform.Compose([
            transform.Crop([args.train_h, args.train_w], crop_type='center', padding=mean, ignore_label=args.ignore_label),
            transform.ToTensor(),
            transform.Normalize(mean=mean, std=std)])
        val_data = dataset.SemData(split='val', data_root=args.data_root, data_list=args.val_list, transform=val_transform)
        if args.distributed:
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
        else:
            val_sampler = None
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val, shuffle=False, num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    max_iou = 0.
    filename = 'PFENet.pth'
    
    for epoch in range(args.start_epoch, args.epochs):
        epoch_log = epoch + 1
        if args.distributed:
            train_sampler.set_epoch(epoch)
        loss_train, mIoU_train, mAcc_train, allAcc_train = train(train_loader, model, t_model, optimizer, epoch, StuTransConv, TeaTransConv, teacher_optimizer)
        if main_process():
            writer.add_scalar('loss_train', loss_train, epoch_log)
            writer.add_scalar('mIoU_train', mIoU_train, epoch_log)
            writer.add_scalar('mAcc_train', mAcc_train, epoch_log)
            writer.add_scalar('allAcc_train', allAcc_train, epoch_log)

        if args.evaluate and epoch % args.eval_freq == 0:
            loss_val, mIoU_val, mAcc_val, allAcc_val = validate(val_loader, model, criterion)
            if main_process():
                writer.add_scalar('loss_val', loss_val, epoch_log)
                writer.add_scalar('mIoU_val', mIoU_val, epoch_log)
                writer.add_scalar('mAcc_val', mAcc_val, epoch_log)
                writer.add_scalar('allAcc_val', allAcc_val, epoch_log)

            if mIoU_val > max_iou and main_process():
                max_iou = mIoU_val
                if os.path.exists(filename):
                    os.remove(filename)            
                filename = args.save_path + '/train_epoch_' + str(epoch) + '_'+str(max_iou)+'.pth'
                logger.info('Saving checkpoint to: ' + filename)
                torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, filename) 

            if epoch > 1 and epoch % args.save_freq == 0 and args.save_freq != 1 and main_process():
                filename = args.save_path + '/train_epoch_' + str(epoch) + '.pth'
                logger.info('Saving checkpoint to: ' + filename)
                torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, filename)                 

    if main_process():
        filename = args.save_path + '/train_epoch_' + str(args.epochs) + '.pth'
        logger.info('Saving checkpoint to: ' + filename)
        torch.save({'epoch': args.epochs, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, filename)                    


def CriterionKD(pred, soft, new_size, raw_soft=None, raw_pred=None, target=None, args=None, cur_step=None, max_step=None):
    '''
    knowledge distillation loss
    '''
    soft.detach()
    h, w = soft.size(2), soft.size(3)
    #raw_pred, raw_soft = pred.clone(), soft.clone()
    if new_size:
        scale_pred = F.interpolate(input=pred, size=(new_size[0], new_size[1]), mode='bilinear', align_corners=True)
        scale_soft = F.interpolate(input=soft, size=(new_size[0], new_size[1]), mode='bilinear', align_corners=True)
        if target is not None and (target.shape[-2] != new_size[0]):
            target = F.interpolate(input=target.float().unsqueeze(1), size=(new_size[0], new_size[1]), mode='nearest').squeeze()        
    else:
        scale_pred = pred
        scale_soft = soft

    loss = torch.nn.KLDivLoss()(F.log_softmax(scale_pred / args.temperature, dim=1), F.softmax(scale_soft / args.temperature, dim=1))
    return loss * args.temperature * args.temperature


def CriterionKmeansInterWhole(pred, soft, target, args, new_size=None, \
                            StuTransConv=None, TeaTransConv=None):
    soft.detach()
    raw_soft = soft.clone()
    if args.stu_trans:
        pred = StuTransConv(pred) 
    if args.tea_trans:
        soft = TeaTransConv(soft)

    batch, c, h, w = soft.shape[:]
    raw_target = target.clone()
    target = F.interpolate(target.unsqueeze(1).float(), size=(h, w), mode='nearest').squeeze(1)

    if args.kmeans_norm:
        soft = F.normalize(soft, 2, 1)
        pred = F.normalize(pred, 2, 1)  
        raw_soft = F.normalize(raw_soft, 2, 1)
    loss_list = []

    tmp_pred = pred.clone()  # c, h, w
    tmp_soft = soft.clone()  # c, h, w
    tmp_target = target.clone() # h, w
    unique_y = list(tmp_target.unique())
    centers_pred_list = []
    centers_soft_list = []
    label_list = []
    tmp_loss_list = []
    for tmp_cls in unique_y:
        if tmp_cls == 255:
            continue
        tmp_mask = (tmp_target == tmp_cls).float()  # b, h, w
        tmp_pos = tmp_mask.nonzero()    # n, 3
        tmp_pred_vec = tmp_pred[tmp_pos[:,0], :, tmp_pos[:,1], tmp_pos[:,2]]  # n, c
        tmp_raw_soft_vec = raw_soft[tmp_pos[:,0], :, tmp_pos[:,1], tmp_pos[:,2]]
        tmp_soft_vec = tmp_soft[tmp_pos[:,0], :, tmp_pos[:,1], tmp_pos[:,2]]   # n, c
        tmp_soft_matrix = tmp_raw_soft_vec @ tmp_raw_soft_vec.permute(1, 0)

        ### GAP
        centers_soft = tmp_soft_vec.mean(0).unsqueeze(0)
        centers_pred = tmp_pred_vec.mean(0).unsqueeze(0)

        if args.kmeans_norm:
            centers_soft = F.normalize(centers_soft, 2, 1)  #  num_clster, c
            centers_pred = F.normalize(centers_pred, 2, 1)  #  num_clster, c
        centers_pred_list.append(centers_pred)
        centers_soft_list.append(centers_soft)
        label_list.append(torch.ones(centers_pred.shape[0]).cuda() * tmp_cls)

    
    if len(centers_pred_list) == 0:
        return torch.zeros(1).cuda(), torch.zeros(1).cuda(), torch.zeros(1).cuda(), torch.zeros(1).cuda()
    else:
        try:
            centers_pred = torch.cat(centers_pred_list, 0).view(-1, centers_pred.shape[1], 1, 1)  # N, c, 1, 1
            centers_soft = torch.cat(centers_soft_list, 0).view(-1, centers_soft.shape[1], 1, 1)  # N, c, 1, 1
            new_labels = torch.cat(label_list, 0).unsqueeze(1)   # N, 1
            new_labels = torch.zeros(new_labels.shape[0], args.classes).cuda().scatter_(1,new_labels.long(),1)   # N, cls 
            new_labels = new_labels.permute(1, 0).view(args.classes, centers_pred.shape[0], 1, 1) # cls, N, 1, 1

            if args.kmeans_norm:
                pred, soft = F.normalize(pred, 2, 1), F.normalize(soft, 2, 1)
                centers_pred, centers_soft = F.normalize(centers_pred, 2, 1), F.normalize(centers_soft, 2, 1)

            pred_out = F.conv2d(input=pred, weight=centers_pred, stride=1, padding=0) 
            soft_out = F.conv2d(input=soft, weight=centers_soft, stride=1, padding=0) 
            sum_soft_out = F.conv2d(input=soft_out, weight=new_labels, stride=1, padding=0)  # b, N, h, w -> b, cls, h, w
            sum_pred_out = F.conv2d(input=pred_out, weight=new_labels, stride=1, padding=0)  # b, N, h, w -> b, cls, h, w
            sum_soft_out = sum_soft_out / (new_labels.sum(1).unsqueeze(0) + 1e-12)
            sum_pred_out = sum_pred_out / (new_labels.sum(1).unsqueeze(0) + 1e-12)    
            if new_size:
                sum_soft_out = F.interpolate(sum_soft_out, size=new_size, mode='bilinear', align_corners=True)
                sum_pred_out = F.interpolate(sum_pred_out, size=new_size, mode='bilinear', align_corners=True)
            else:
                raw_target = target.clone()

            pred_out = pred_out * args.kmeans_temp
            soft_out = soft_out * args.kmeans_temp
            sum_soft_out = sum_soft_out * args.kmeans_temp
            sum_pred_out = sum_pred_out * args.kmeans_temp

            soft_loss = nn.CrossEntropyLoss(ignore_index=255)(sum_soft_out, raw_target.long())
            pred_kl_loss = torch.nn.KLDivLoss()(F.log_softmax(pred_out, dim=1), F.softmax(soft_out.detach(), dim=1)) 

            ### get losses with centers
            centers_pred = centers_pred.squeeze(-1).squeeze(-1) # n, c
            centers_soft = centers_soft.squeeze(-1).squeeze(-1).detach()

            ### proto_align_loss
            proto_matrix = centers_pred * centers_soft   #  n, c
            proto_matrix = proto_matrix.sum(1)  # n
            proto_align_loss = 0.5 * (1 - proto_matrix).mean()

            return pred_kl_loss, soft_loss, proto_align_loss

        except Exception as e:
            if main_process():
                logger.info(e)
            return torch.zeros(1).cuda(), torch.zeros(1).cuda(), torch.zeros(1).cuda()


def train(train_loader, model, t_model, optimizer, epoch, StuTransConv, TeaTransConv, teacher_optimizer):
    torch.cuda.empty_cache()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    main_loss_meter = AverageMeter()
    aux_loss_meter = AverageMeter()
    KL_loss_meter = AverageMeter()
    soft_kmeans_loss_meter =AverageMeter()
    proto_align_loss_meter = AverageMeter()
    pred_kmeans_loss_meter = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.train()
    if t_model is not None:
        t_model.eval()
    StuTransConv.train()
    TeaTransConv.train()
    end = time.time()
    max_iter = args.epochs * len(train_loader)
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        if args.zoom_factor != 8:
            h = int((target.size()[1] - 1) / 8 * args.zoom_factor + 1)
            w = int((target.size()[2] - 1) / 8 * args.zoom_factor + 1)
            # 'nearest' mode doesn't support align_corners mode and 'bilinear' mode is fine for downsampling
            target = F.interpolate(target.unsqueeze(1).float(), size=(h, w), mode='bilinear', align_corners=True).squeeze(1).long()
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        ## check input validity
        small_target = F.interpolate(target.unsqueeze(1).float(), size=(60, 60), mode='bilinear', align_corners=True).squeeze(1).long()
        unique_small_labels = list(small_target.unique())
        if 255 in unique_small_labels:
            unique_small_labels.remove(255)
        if len(unique_small_labels) == 0:
            if main_process():
                logger.info('Skip with invalid input.')
            continue

        cur_step = i + epoch * (len(train_loader))
        s_output, main_loss, aux_loss, s_feats = model(input, target)

        kl_loss = torch.zeros_like(main_loss)
        if t_model is not None:
            with torch.no_grad():
                t_output, _, _, t_feats = t_model(input, target)          
        if args.kl_weight > 0:
            ## compute kl loss  
            kl_loss = CriterionKD(pred=s_feats[-1], soft=t_feats[-1], new_size=(target.shape[-2], target.shape[-1]), raw_pred=s_feats[-1], raw_soft=t_feats[-1], \
                                    target=target, args=args, cur_step=cur_step, max_step=max_iter)


        pred_kmeans_loss, soft_kmeans_loss = torch.zeros_like(main_loss), torch.zeros_like(main_loss)
        proto_align_loss = torch.zeros_like(main_loss)
        if args.pred_kmeans_weight > 0:
            pred_kmeans_loss, soft_kmeans_loss, proto_align_loss = CriterionKmeansInterWhole(pred=s_feats[-2], soft=t_feats[-2].detach(), target=target, args=args, new_size=(target.shape[-2], target.shape[-1]), \
                                                StuTransConv=StuTransConv, TeaTransConv=TeaTransConv)

        if args.tea_trans and args.pred_kmeans_weight and soft_kmeans_loss.sum() > 0:
            teacher_optimizer.zero_grad()
            soft_kmeans_loss.backward()
            teacher_optimizer.step()

        if not args.multiprocessing_distributed:
            main_loss, aux_loss = torch.mean(main_loss), torch.mean(aux_loss)
            kl_loss = torch.mean(kl_loss)
        
        loss = args.main_weight * main_loss + args.aux_weight * aux_loss \
               + args.kl_weight * kl_loss \
               + args.pred_kmeans_weight * pred_kmeans_loss \
               + args.proto_align_weight * proto_align_loss \

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        n = input.size(0)
        if args.multiprocessing_distributed:
            main_loss, aux_loss, loss = main_loss.detach() * n, aux_loss * n, loss * n  # not considering ignore pixels
            count = target.new_tensor([n], dtype=torch.long)
            dist.all_reduce(main_loss), dist.all_reduce(aux_loss), dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            main_loss, aux_loss, loss = main_loss / n, aux_loss / n, loss / n

        intersection, union, target = intersectionAndUnionGPU(s_output, target, args.classes, args.ignore_label)
        if args.multiprocessing_distributed:
            dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        mean_accuracy = sum(intersection_meter.avg) / (sum(target_meter.avg) + 1e-10)
        main_loss_meter.update(main_loss.item(), n)
        aux_loss_meter.update(aux_loss.item(), n)
        KL_loss_meter.update(kl_loss.item(), n)
        pred_kmeans_loss_meter.update(pred_kmeans_loss.item(), n)
        soft_kmeans_loss_meter.update(soft_kmeans_loss.item(), n)
        proto_align_loss_meter.update(proto_align_loss.item(), n)
        loss_meter.update(loss.item(), n)
        batch_time.update(time.time() - end)
        end = time.time()

        current_iter = epoch * len(train_loader) + i + 1
        current_lr = poly_learning_rate(args.base_lr, current_iter, max_iter, power=args.power)
        for index in range(0, args.index_split):
            optimizer.param_groups[index]['lr'] = current_lr
        for index in range(args.index_split, len(optimizer.param_groups)):
            optimizer.param_groups[index]['lr'] = current_lr * 10

        current_iter = epoch * len(train_loader) + i + 1
        current_lr = poly_learning_rate(args.teacher_lr, current_iter, max_iter, power=args.power)
        for index in range(args.index_split, len(teacher_optimizer.param_groups)):
            teacher_optimizer.param_groups[index]['lr'] = current_lr 

        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'MainLoss {main_loss_meter.val:.4f} ({main_loss_meter.avg:.4f}) '
                        'AuxLoss {aux_loss_meter.val:.4f} '
                        'KLLoss {KL_loss_meter.val:.4f} ({KL_loss_meter.avg:.4f}) '
                        'SoftKM  {soft_kmeans_loss_meter.val:.4f} ({soft_kmeans_loss_meter.avg:.4f}) '
                        'PredKM {pred_kmeans_loss_meter.val:.4f} ({pred_kmeans_loss_meter.avg:.4f}) '
                        'PAlign {proto_align_loss_meter.val:.4f} ({proto_align_loss_meter.avg:.4f}) '
                        'Loss {loss_meter.val:.4f} '
                        'Accuracy {accuracy:.4f} ({mean_accuracy:.4f}).'.format(epoch+1, args.epochs, i + 1, len(train_loader),
                                                          batch_time=batch_time,
                                                          data_time=data_time,
                                                          remain_time=remain_time,
                                                          main_loss_meter=main_loss_meter,
                                                          aux_loss_meter=aux_loss_meter,
                                                          KL_loss_meter=KL_loss_meter,
                                                          soft_kmeans_loss_meter=soft_kmeans_loss_meter,
                                                          proto_align_loss_meter=proto_align_loss_meter,
                                                          pred_kmeans_loss_meter=pred_kmeans_loss_meter,
                                                          loss_meter=loss_meter,
                                                          accuracy=accuracy,
                                                          mean_accuracy=mean_accuracy))
        if main_process():
            writer.add_scalar('loss_train_batch', main_loss_meter.val, current_iter)
            writer.add_scalar('mIoU_train_batch', np.mean(intersection / (union + 1e-10)), current_iter)
            writer.add_scalar('mAcc_train_batch', np.mean(intersection / (target + 1e-10)), current_iter)
            writer.add_scalar('allAcc_train_batch', accuracy, current_iter)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if main_process():
        logger.info('Train result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(epoch+1, args.epochs, mIoU, mAcc, allAcc))
    return main_loss_meter.avg, mIoU, mAcc, allAcc


def validate(val_loader, model, criterion):
    torch.cuda.empty_cache()
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        data_time.update(time.time() - end)
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        with torch.no_grad():
            output = model(input)
        if args.zoom_factor != 8:
            output = F.interpolate(output, size=target.size()[1:], mode='bilinear', align_corners=True)
        loss = criterion(output, target)

        n = input.size(0)
        if args.multiprocessing_distributed:
            loss = loss * n  # not considering ignore pixels
            count = target.new_tensor([n], dtype=torch.long)
            dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            loss = loss / n
        else:
            loss = torch.mean(loss)

        output = output.max(1)[1]
        intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        if args.multiprocessing_distributed:
            dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        loss_meter.update(loss.item(), input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if ((i + 1) % args.print_freq == 0) and main_process():
            logger.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                        'Accuracy {accuracy:.4f}.'.format(i + 1, len(val_loader),
                                                          data_time=data_time,
                                                          batch_time=batch_time,
                                                          loss_meter=loss_meter,
                                                          accuracy=accuracy))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if main_process():
        logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
        for i in range(args.classes):
            logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')
    return loss_meter.avg, mIoU, mAcc, allAcc


if __name__ == '__main__':
    main()

