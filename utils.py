import os
import sys
import math
import json
import torch
import random
import logging

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.distributed as dist

from PIL import Image, ImageEnhance, ImageOps
from tqdm import tqdm
from thop import profile
from torch.optim.lr_scheduler import LambdaLR


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# --------------------------------------------- con_loss ------------------------------------------------------ #
def con_loss(features, labels):
    B, _ = features.shape   # [B, 768], labels = [B*1]
    features = F.normalize(features)    
    cos_matrix = features.mm(features.t())     
    pos_label_matrix = torch.stack([labels == labels[i] for i in range(B)]).float() 
    neg_label_matrix = 1 - pos_label_matrix  
    pos_cos_matrix = 1 - cos_matrix     
    neg_cos_matrix = cos_matrix - 0.4   
    neg_cos_matrix[neg_cos_matrix < 0] = 0
    loss = (pos_cos_matrix * pos_label_matrix).sum() + (neg_cos_matrix * neg_label_matrix).sum()
    loss /= (B * B)
    return loss


# --------------------------------------------- RLM ---------------------------------------------------- #
def get_mask_img(images, attentions, device, mode='crop', theta=0.5, padding_ratio=0.1):
    result = torch.eye(attentions[0].size(-1)).to(device)
    # attentions = attentions.to("cpu")
    with torch.no_grad():
        for attention in attentions:
            attention_heads_fused = attention.max(axis=1)[0]
            # print("attention_heads_fused.shape:", attention_heads_fused.shape)  # [B, 12, 785, 785]->[B, 785, 785]
            I = torch.eye(attention_heads_fused.size(-1)).to(device)
            # print("I:", I.shape)    # [785, 785]
            a = (attention_heads_fused + 1.0 * I) / 2
            # print("a.shape:", a.shape)
            # print("a.sum.shape:", a.sum(dim=-1).shape)
            # a = a / a.sum(dim=-1)
            # print("a:", a.shape)    # [B, 785, 785]
            result = torch.matmul(a, result)

    mask = result[:, 0, 1:]
 
    # In case of 224x224 image, this brings us from 196 to 14
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
            # print("crop_mask.shape:", crop_mask.shape)  # [1, 1, 448, 448]
            nonzero_indices = torch.nonzero(crop_mask[0, 0, ...])
            # print("nonzero_indices.shape:", nonzero_indices.shape)
            height_min = max(int(nonzero_indices[:, 0].min().item() - padding_ratio * imgH), 0)
            height_max = min(int(nonzero_indices[:, 0].max().item() + padding_ratio * imgH), imgH)
            width_min = max(int(nonzero_indices[:, 1].min().item() - padding_ratio * imgW), 0)
            width_max = min(int(nonzero_indices[:, 1].max().item() + padding_ratio * imgW), imgW)
            
            crop_images.append(
                F.interpolate(images[batch_index:batch_index + 1, :, height_min:height_max, width_min:width_max],
                                    size=(imgH, imgW)))
        
        crop_images = torch.cat(crop_images, dim=0)
        # print("crop_images.shape:", crop_images.shape)  # [12, 3, 448, 448]
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


# --------------------------------------------- train ------------------------------------------------------ #
def train_one_epoch(model, args, optimizer, data_loader, device, epoch, lr_scheduler):
    model.train()
    if args.label_smoothing == 0 :
        loss_function = torch.nn.CrossEntropyLoss()
    else:
        loss_function = LabelSmoothing(args.label_smoothing)
        
    accu_loss = torch.zeros(1).to(device)   
    accu_num = torch.zeros(1).to(device)    
    accu_conloss = torch.zeros(1).to(device)  
    accu_croploss = torch.zeros(1).to(device)   
    accu_droploss = torch.zeros(1).to(device)   
    optimizer.zero_grad()

    sample_num = 0
    accum_iter = 4  
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
            # print("attn_weights.len:", len(attn_weights))   # [12]
            # print("attn_weight[0].shape:", attn_weights[0].shape)   # [16, 12, 785, 785]
            with torch.no_grad():
                crop_images = get_mask_img(images, attn_weights, device, mode="crop", theta=(0.4, 0.6), padding_ratio=0.1)
                drop_images = get_mask_img(images, attn_weights, device, mode="drop", theta=(0.2, 0.5))
            # print("crop_images.shape:", crop_images.shape)
            
            # crop images forward
            y_pred_crop, crop_feature, _ = model(crop_images.to(device))
            # drop images forward
            y_pred_drop, drop_feature, _ = model(drop_images.to(device))
            
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels).sum()

        if args.use_loss is True:
            if not args.use_attn:               
                c_loss = con_loss(x_feature, labels)
                loss = loss_function(pred, labels) + c_loss
                accu_conloss += c_loss
            else:
                x_feature = (x_feature + crop_feature + drop_feature) / 3.
                
                c_loss = con_loss(x_feature, labels)
                crop_loss = loss_function(y_pred_crop, labels)
                drop_loss = loss_function(y_pred_drop, labels)
                
                loss = loss_function(pred, labels) /3. + \
                crop_loss / 3. + \
                drop_loss / 3. + c_loss
                
                accu_conloss += c_loss
                accu_croploss += crop_loss
                accu_droploss += drop_loss
        else:
            if not args.use_attn:
                loss = loss_function(pred, labels)
            else:
                crop_loss = loss_function(y_pred_crop, labels)
                drop_loss = loss_function(y_pred_drop, labels)
                
                loss = loss_function(pred, labels) /3. + \
                crop_loss / 3. + \
                drop_loss / 3.
                
                accu_droploss += drop_loss
            
        if args.batch_accum:    
            loss = loss / accum_iter
        loss.backward()
        accu_loss += loss.detach()  

        data_loader.desc = "[train epoch {}] loss : {:.4f}, lr: {:.4f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num,
            optimizer.param_groups[0]["lr"]
        )

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training', loss)
            sys.exit(1)

        if args.batch_accum:
            if ((step + 1) % accum_iter == 0) or (step + 1 == len(data_loader)):
                optimizer.step()
                optimizer.zero_grad()
        else:
            optimizer.step()
            optimizer.zero_grad()
            # update lr
            lr_scheduler.step()

    logging.info('[train epoch {}] loss: {:.4f}, con_loss: {:.4f}, crop_loss: {:.4f}, drop_loss: {:.4f}, acc: {:.4f}'.format(epoch,
                                                                                       accu_loss.item() / (step + 1), 
                                                                                       accu_conloss.item() / (step + 1),
                                                                                       accu_croploss.item() / (step + 1),
                                                                                       accu_droploss.item() / (step + 1),
                                                                                       accu_num.item() / sample_num))
    logging.info('')
    
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num

# --------------------------------------------- valid ------------------------------------------------------ #
@torch.no_grad()
def evaluate(model, args, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu = torch.zeros(1).to(device)   
    accu_loss = torch.zeros(1).to(device)  
    accu_conloss = torch.zeros(1).to(device)  
    
    sample_num = 0
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
                crop_images = get_mask_img(images, attn_weights, device, mode="crop", theta=0.1, padding_ratio=0.05)   
            y_pred_crop, crop_feature, _ = model(crop_images)   
        
        pred_classes = torch.max(pred, dim=1)[1]
        accu += torch.eq(pred_classes, labels).sum()

        if args.use_loss is True:
            if not args.use_attn:
                c_loss = con_loss(x_feature, labels)
                loss = loss_function(pred, labels) + c_loss
                accu_conloss += c_loss
            else:
                x_feature = (x_feature + crop_feature) / 2.
                pred = (pred + y_pred_crop) / 2.
                c_loss = con_loss(x_feature, labels)
                loss = loss_function(pred, labels) + c_loss
                accu_conloss += c_loss
        else:
            if not args.use_attn:
                loss = loss_function(pred, labels)
            else:
                pred = (pred + y_pred_crop) / 2.
                loss = loss_function(pred, labels)
            
        accu_loss += loss
        
        data_loader.desc = "[valid epoch {}] loss: {:.4f}, acc: {:.4f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu.item() / sample_num
        )

    logging.info('[valid epoch {}] loss: {:.4f}, con_loss: {:.4f}, acc: {:.4f}'.format(epoch, accu_loss.item() / (step + 1), accu_conloss.item() / (step + 1),accu.item() / sample_num))
    logging.info('')

    return accu_loss.item() / (step + 1), accu.item() / sample_num

# -------------------------------------------- topk acc ------------------------------------------------- #
def accuracy(output, target, topk=(5,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)  
    pred = pred.t()  
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

# -------------------------------------------- random seed ------------------------------------------------- #
def set_seed(seed):

    np.random.seed(seed) 
    random.seed(seed)
    
    os.environ['PYTHONHASHSEED'] = str(seed)  
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  
    
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.manual_seed(seed)
    
    torch.use_deterministic_algorithms(True) 
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.enabled = False  
    torch.backends.cudnn.benchmark = False 

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class SubPolicy(object):
    def __init__(self, p1, operation1, magnitude_idx1, p2, operation2, magnitude_idx2, fillcolor=(128, 128, 128)):
        ranges = {
            "shearX": np.linspace(0, 0.3, 10),
            "shearY": np.linspace(0, 0.3, 10),
            "translateX": np.linspace(0, 150 / 331, 10),
            "translateY": np.linspace(0, 150 / 331, 10),
            "rotate": np.linspace(0, 30, 10),
            "color": np.linspace(0.0, 0.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(np.int),
            "solarize": np.linspace(256, 0, 10),
            "contrast": np.linspace(0.0, 0.9, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.9, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "invert": [0] * 10
        }

        def rotate_with_fill(img, magnitude):
            rot = img.convert("RGBA").rotate(magnitude)
            return Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot).convert(img.mode)

        func = {
            "shearX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "shearY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "translateX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0),
                fillcolor=fillcolor),
            "translateY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * img.size[1] * random.choice([-1, 1])),
                fillcolor=fillcolor),
            "rotate": lambda img, magnitude: rotate_with_fill(img, magnitude),
            # "rotate": lambda img, magnitude: img.rotate(magnitude * random.choice([-1, 1])),
            "color": lambda img, magnitude: ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([-1, 1])),
            "posterize": lambda img, magnitude: ImageOps.posterize(img, magnitude),
            "solarize": lambda img, magnitude: ImageOps.solarize(img, magnitude),
            "contrast": lambda img, magnitude: ImageEnhance.Contrast(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "sharpness": lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "brightness": lambda img, magnitude: ImageEnhance.Brightness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "autocontrast": lambda img, magnitude: ImageOps.autocontrast(img),
            "equalize": lambda img, magnitude: ImageOps.equalize(img),
            "invert": lambda img, magnitude: ImageOps.invert(img)
        }

        # self.name = "{}_{:.2f}_and_{}_{:.2f}".format(
        #     operation1, ranges[operation1][magnitude_idx1],
        #     operation2, ranges[operation2][magnitude_idx2])
        self.p1 = p1
        self.operation1 = func[operation1]
        self.magnitude1 = ranges[operation1][magnitude_idx1]
        self.p2 = p2
        self.operation2 = func[operation2]
        self.magnitude2 = ranges[operation2][magnitude_idx2]

    def __call__(self, img):
        if random.random() < self.p1:
            img = self.operation1(img, self.magnitude1)
        if random.random() < self.p2:
            img = self.operation2(img, self.magnitude2)
        return img
    

class AutoAugImageNetPolicy(object):
    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.4, "posterize", 8, 0.6, "rotate", 9, fillcolor),
            SubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.6, "equalize", 3, fillcolor),
            SubPolicy(0.6, "posterize", 7, 0.6, "posterize", 6, fillcolor),
            SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4, fillcolor),

            SubPolicy(0.4, "equalize", 4, 0.8, "rotate", 8, fillcolor),
            SubPolicy(0.6, "solarize", 3, 0.6, "equalize", 7, fillcolor),
            SubPolicy(0.8, "posterize", 5, 1.0, "equalize", 2, fillcolor),
            SubPolicy(0.2, "rotate", 3, 0.6, "solarize", 8, fillcolor),
            SubPolicy(0.6, "equalize", 8, 0.4, "posterize", 6, fillcolor),

            SubPolicy(0.8, "rotate", 8, 0.4, "color", 0, fillcolor),
            SubPolicy(0.4, "rotate", 9, 0.6, "equalize", 2, fillcolor),
            SubPolicy(0.0, "equalize", 7, 0.8, "equalize", 8, fillcolor),
            SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, fillcolor),
            SubPolicy(0.6, "color", 4, 1.0, "contrast", 8, fillcolor),

            SubPolicy(0.8, "rotate", 8, 1.0, "color", 2, fillcolor),
            SubPolicy(0.8, "color", 8, 0.8, "solarize", 7, fillcolor),
            SubPolicy(0.4, "sharpness", 7, 0.6, "invert", 8, fillcolor),
            SubPolicy(0.6, "shearX", 5, 1.0, "equalize", 9, fillcolor),
            SubPolicy(0.4, "color", 0, 0.6, "equalize", 3, fillcolor),

            SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4, fillcolor),
            SubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5, fillcolor),
            SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, fillcolor),
            SubPolicy(0.6, "color", 4, 1.0, "contrast", 8, fillcolor)
        ]

    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment ImageNet Policy"


# -------------------------------------------- collate_fn ------------------------------------------------- #

def collate_fn(batch):
    # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
    images, labels = tuple(zip(*batch))
    images = images.ToTensor()
    images = torch.stack(images, dim=0)
    
    return images, labels

# -------------------------------------------- data transform------------------------------------------------- #
def get_transform(resize, phase='train'):
    # int(resize[0] / 0.875)
    if resize == 448:
        new_size = 550
    elif resize == 304:
        new_size = 400
    
    if resize == 304:
        if phase == 'train':
            return transforms.Compose([transforms.Resize((400, 400), Image.BILINEAR),
                                    transforms.RandomCrop((304, 304)),
                                    transforms.RandomHorizontalFlip(),
                                    AutoAugImageNetPolicy(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        else:
            return transforms.Compose([transforms.Resize((400, 400), Image.BILINEAR),
                                    transforms.CenterCrop((304, 304)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
     
    else:  
        if phase == 'train':
            return transforms.Compose([
            # size=(int(resize[0] / 0.875), int(resize[0] / 0.875))
            transforms.Resize((new_size, new_size)),
            transforms.RandomCrop(resize),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.126, saturation=0.5),   
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        else:
            return transforms.Compose([
            transforms.Resize((new_size, new_size)),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

# -------------------------------------------- load weights------------------------------------------------- #
def load_weights(args, model):

    # ViT(448*448)
    if args.weights != "":
        assert os.path.exists(args.weights), "model weights {}  not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=torch.device(args.device))
        
        pos_embed = weights_dict['pos_embed']
        new_size = args.image_size
        src_shape = (14, 14)    
        dst_shape = (14*new_size/224, 14*new_size/224)  

        num_extra_tokens = 1    
        
        _, L, C = pos_embed.shape
        src_h, src_w = src_shape

        assert L == src_h * src_w + num_extra_tokens, "pos_embed dimmension error."


        extra_tokens = pos_embed[:, :num_extra_tokens] 
        src_weight = pos_embed[:, num_extra_tokens:] 

        src_weight = src_weight.reshape(1, src_h, src_w, C).permute(0, 3, 1, 2) 

        dst_weight = F.interpolate(src_weight, scale_factor=new_size/224, mode='bicubic', align_corners=False) 
 
        dst_weight = torch.flatten(dst_weight, 2).transpose(1, 2) 
        pos_embed = torch.cat((extra_tokens, dst_weight), dim=1) 
        
        weights_dict['pos_embed'] = pos_embed
        
        del_keys = ['head.weight', 'head.bias'] if model.has_logits \
            else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
        for k in del_keys:
            del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            if "head" not in name and "pre_logits" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))
    return model


def get_params_groups(model: torch.nn.Module, weight_decay: float = 1e-5):

    parameter_group_vars = {"decay": {"params": [], "weight_decay": weight_decay},
                            "no_decay": {"params": [], "weight_decay": 0.}}


    parameter_group_names = {"decay": {"params": [], "weight_decay": weight_decay},
                             "no_decay": {"params": [], "weight_decay": 0.}}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        if len(param.shape) == 1 or name.endswith(".bias"):
            group_name = "no_decay"
        else:
            group_name = "decay"

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)

    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000


def count_flops(model, args):
    input = torch.randn(1, 3, args.image_size, args.image_size)
    flops, params = profile(model, inputs=(input, ))
    return flops/1000000000, params/1000000


class WarmupLinearSchedule(LambdaLR):
    """ Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
    """
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))


class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))

class CosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    def __init__(self, optimizer, t_total, cycles=.5, last_epoch=-1):
        self.t_total = t_total
        self.cycles = cycles
        super(CosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        # progress after warmup
        progress = float(step) / float(max(1, self.t_total))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))
    

def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        freeze_epoch=0,
                        freeze_lr=1e-3,
                        warmup_factor=1e-3,
                        end_factor=1e-5):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):

        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            # current_step = (x - num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            # 1 -> end_factor
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
    

# --------------------------------------- LabelSmoothing  --------------------------------------------- #
class LabelSmoothing(nn.Module):
    """NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        """Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
