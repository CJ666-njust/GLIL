import os
import sys
import logging

import random
import torch
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from utils import con_loss


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode.')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'  
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    dist.barrier()  


def get_mask_img(images, attentions, device, mode='crop', theta=0.5, padding_ratio=0.1):
    result = torch.eye(attentions[0].size(-1)).to(device)

    with torch.no_grad():
        for attention in attentions:
            attention_heads_fused = attention.max(axis=1)[0]         
            I = torch.eye(attention_heads_fused.size(-1)).to(device)
            a = (attention_heads_fused + 1.0 * I) / 2
            result = torch.matmul(a, result)

    mask = result[:, 0, 1:]

    width = int(mask.size(-1) ** 0.5)
    mask = torch.reshape(mask, (mask.size(0), width, width))
    mask = mask / torch.max(mask)
    mask = mask.unsqueeze(dim=1)

    
    batches, _, imgH, imgW = images.size()
    if mode == 'crop':
        crop_images = []
        for batch_index in range(batches):
            atten_map = mask[batch_index:batch_index + 1]
            if isinstance(theta, tuple):
                theta_c = random.uniform(*theta) * atten_map.max()
            else:
                theta_c = theta * atten_map.max()

            crop_mask = F.interpolate(atten_map, size=(imgH, imgW)) >= theta_c

            nonzero_indices = torch.nonzero(crop_mask[0, 0, ...])

            height_min = max(int(nonzero_indices[:, 0].min().item() - padding_ratio * imgH), 0)
            height_max = min(int(nonzero_indices[:, 0].max().item() + padding_ratio * imgH), imgH)
            width_min = max(int(nonzero_indices[:, 1].min().item() - padding_ratio * imgW), 0)
            width_max = min(int(nonzero_indices[:, 1].max().item() + padding_ratio * imgW), imgW)
            
            crop_images.append(
                F.interpolate(images[batch_index:batch_index + 1, :, height_min:height_max, width_min:width_max],
                                    size=(imgH, imgW)))
        
        crop_images = torch.cat(crop_images, dim=0)
        return crop_images
    
    elif mode == 'drop':
        drop_masks = []
        for batch_index in range(batches):
            atten_map = mask[batch_index:batch_index + 1]
            if isinstance(theta, tuple):
                theta_d = random.uniform(*theta) * atten_map.max()
            else:
                theta_d = theta * atten_map.max()

            drop_masks.append(F.interpolate(atten_map, size=(imgH, imgW)) < theta_d)
        drop_masks = torch.cat(drop_masks, dim=0).to(device)
        drop_images = images.to(device) * drop_masks.float()
        return drop_images

    else:
        raise ValueError('Expected mode in [\'crop\', \'drop\'], but received unsupported augmentation method %s' % mode)


# --------------------------------------------- 训练 ------------------------------------------------------ #
def train_one_epoch(model, args, optimizer, data_loader, device, epoch, lr_scheduler):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)   
    accu_num = torch.zeros(1).to(device)   
    accu_conloss = torch.zeros(1).to(device)  
    accu_croploss = torch.zeros(1).to(device)  
    accu_droploss = torch.zeros(1).to(device)   
    optimizer.zero_grad()

    sample_num = 0
    if is_main_process():
        data_loader = tqdm(data_loader, file=sys.stdout)
  
    for step, data in enumerate(data_loader):
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        sample_num += images.shape[0]

        if not args.use_attn:
            pred, x_feature = model(images)
        else:
            pred, x_feature, attn_weights = model(images)

            with torch.no_grad():
                crop_images = get_mask_img(images, attn_weights, device, mode="crop", theta=args.crop_rate, padding_ratio=0.1)
                # drop_images = get_mask_img(images, attn_weights, device, mode="drop", theta=(0.2, 0.5))
            
            # crop images forward
            y_pred_crop, crop_feature, _ = model(crop_images.to(device))
            # drop images forward
            # y_pred_drop, drop_feature, _ = model(drop_images.to(device))
            
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels).sum()
        
            
        if args.use_loss is True:
            if not args.use_attn:               
                c_loss = con_loss(x_feature, labels)
                loss = loss_function(pred, labels) + c_loss
                # accu_conloss += c_loss
            else:
                # x_feature = (x_feature + crop_feature + drop_feature) / 3.
                
                c_loss = con_loss(x_feature, labels)
                crop_loss = loss_function(y_pred_crop, labels)
                # drop_loss = loss_function(y_pred_drop, labels)
                
                loss = loss_function(pred, labels) / 2. + \
                crop_loss / 2. + \
                c_loss
                
        else:
            if not args.use_attn:
                loss = loss_function(pred, labels)
            else:
                crop_loss = loss_function(y_pred_crop, labels)
                # drop_loss = loss_function(y_pred_drop, labels)
                
                loss = loss_function(pred, labels) /2. + \
                crop_loss / 2. 
                # drop_loss / 3.
                
        loss.backward()
        loss = reduce_value(loss, average=False)

        accu_loss = (accu_loss * step + loss.detach()) / (step + 1)  
        
        if args.use_loss is True:
            accu_conloss = (accu_conloss * step + c_loss.detach()) / (step + 1)
        if args.use_attn is True:
            accu_croploss = (accu_croploss * step + crop_loss.detach()) / (step + 1)
        # accu_droploss = (accu_droploss * step + drop_loss.detach()) / (step + 1)
        
        # print
        if is_main_process():
            data_loader.desc = "[train epoch {}] loss : {:.4f}, acc :{:.4f}, lr: {:.5f}".format(
                epoch,
                accu_loss.item(),
                accu_num.item() / sample_num,
                optimizer.param_groups[0]["lr"]
            )

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training', loss)
            sys.exit(1)

        optimizer.step()
        # optimizer.zero_grad()
        # update lr
        lr_scheduler.step()

        if device != torch.device("cpu"):
            torch.cuda.synchronize(device)

    if is_main_process():
        logging.info('[train epoch {}] loss: {:.4f}, con_loss: {:.4f}, crop_loss: {:.4f}, drop_loss: {:.4f}, acc: {:.4f}'.format(epoch,
                                                                                       accu_loss.item() / (step + 1), 
                                                                                       accu_conloss.item() / (step + 1),
                                                                                       accu_croploss.item() / (step + 1),
                                                                                       accu_droploss.item() / (step + 1),
                                                                                       accu_num.item() / sample_num))
    logging.info('')
        
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


# --------------------------------------------- evaluate ------------------------------------------------------ #
@torch.no_grad()
def evaluate(model, args, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()
    model.eval()

    accu_top1 = torch.zeros(1).to(device)   
    accu_two = torch.zeros(1).to(device)
    accu_loss = torch.zeros(1).to(device)  
    accu_conloss = torch.zeros(1).to(device)  
    # accu_croploss = torch.zeros(1).to(device)  
    
    sample_num = 0
    if is_main_process():
        data_loader = tqdm(data_loader, file=sys.stdout)
        
    for step, data in enumerate(data_loader):
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        sample_num += images.shape[0]

        if not args.use_attn:
            pred, x_feature = model(images)
        else:
            pred, x_feature, attn_weights = model(images) 
            with torch.no_grad():
                crop_images = get_mask_img(images, attn_weights, device, mode="crop", theta=args.pred_crop_rate, padding_ratio=0.05)   
            y_pred_crop, crop_feature, _ = model(crop_images)
           
        pred_classes = torch.max(pred, dim=1)[1]
        accu_top1 += torch.eq(pred_classes, labels).sum()
               
        if args.use_loss is True:
            if not args.use_attn:
                c_loss = con_loss(x_feature, labels)
                loss = loss_function(pred, labels) + c_loss
                
            else:
                # x_feature = (x_feature + crop_feature) / 2.
                pred = (pred + y_pred_crop) / 2.
                c_loss = con_loss(x_feature, labels)
                loss = loss_function(pred, labels) + c_loss
                
        else:
            if not args.use_attn:
                loss = loss_function(pred, labels)
            else:
                pred = (pred + y_pred_crop) / 2.
                loss = loss_function(pred, labels)
            
        accu_loss += loss
        if args.use_loss:
            accu_conloss += c_loss
            
        pred_classes = torch.max(pred, dim=1)[1]
        accu_two += torch.eq(pred_classes, labels).sum()
        

        if is_main_process():
            data_loader.desc = "[valid epoch {}] loss: {:.4f}, acc: {:.4f}".format(
                epoch,
                accu_loss.item() / (step + 1),
                accu_top1.item() / sample_num
            )

    if is_main_process():
        logging.info('[valid epoch {}] loss: {:.4f}, con_loss: {:.4f}, acc: {:.4f}, acc_two: {:.4f}'.format(epoch, accu_loss.item() / (step + 1), accu_conloss.item() / (step + 1),accu_top1.item() / sample_num,accu_two.item() / sample_num))
        logging.info('')

    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    # accu_loss = reduce_value(accu_loss, average=False)
    # accu_top1 = reduce_value(accu_top1, average=False)    
    
    return accu_loss.item() / (step + 1), max(accu_top1.item(), accu_two.item()) / sample_num


def cleanup():
    dist.destroy_process_group()


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def reduce_value(value, average=True):
    world_size = get_world_size()
    if world_size < 2:  # single GPU
        return value

    with torch.no_grad():
        dist.all_reduce(value)
        if average:
            value /= world_size

        return value