import os
import math
import time
import tempfile
import argparse
import logging
import warnings

import torch
import torch.optim as optim
import torch.distributed as dist
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms


from model.change_vit import vit_base_patch16_224_in21k

from datasets import get_trainval_datasets
from utils import set_seed, seed_worker, get_params_groups, WarmupCosineSchedule, load_weights, create_lr_scheduler
    
from distributed_utils import init_distributed_mode, train_one_epoch, evaluate,  cleanup

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # change gpu numbering order

attn = True

def get_args_parser():
    parser = argparse.ArgumentParser('hyper parameter for image classification', add_help=False)


    parser.add_argument('--use_attn', default=True, type=bool)
    parser.add_argument('--use_loss', default=False, type=bool)
    
    parser.add_argument('--crop_rate', default=0.6, type=float)
    parser.add_argument('--pred_crop_rate', default=0.1, type=float)

    parser.add_argument('--device', default='cuda')
    parser.add_argument('--nw', default=8)
    

    parser.add_argument('--data_path', default=r"dataset/CUB_200_2011", type=str)
    parser.add_argument('--tag', default='bird', type=str,
                        help='available tags include bird, car, dog and aircraft.')
    parser.add_argument('--image_size', default=448, type=int)


    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--epochs', default=120, type=int)
    parser.add_argument('--warmup', default=False, type=bool)
    parser.add_argument('--warmup_epochs', default=20, type=int)

 
    parser.add_argument('--weights', default="pre_trained/vit_base_patch16_224_in21k.pth", type=str)
    parser.add_argument('--freeze_layers', default=False, type=bool)  
    parser.add_argument('--freeze_epoch', default=0, type=int)    
    parser.add_argument('--freeze_lr', default=1e-3, type=float)    

    parser.add_argument('--save_weights', default="save_results", type=str)  
    parser.add_argument('--save_name', default="best_model.pth", type=str)  
    parser.add_argument('--tensorboard_dir', default="save_results/runs", help='path where to tensorboard log.')
    parser.add_argument('--log_name', default='train_log.txt', type=str)


    parser.add_argument('--world-size', default=4, type=int, help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--syncBN', type=bool, default=False)
    

    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--weights_decay', default=5e-5, type=float)
    parser.add_argument('--momentum', default=0.9, type=float, help='SGD momentum (default: 0.9).')


    parser.add_argument('--seed', default=14, type=int)

    return parser


def main(args):
    # -------------------------------------- init ------------------------------------------- #
    # Synchronize time across servers
    timezone = time.strftime('%Z', time.localtime()) # UTC
    if timezone == "UTC":
        args.data_path = "/data/cj/" + args.data_path
    else:
        args.data_path = "/data1/chenjin/" + args.data_path
    
    if 'dog' in args.data_path:
        args.tag = 'dog'
    elif 'car' in args.data_path:
        args.tag = 'car'
    elif 'aircraft' in args.data_path:
        args.tag = 'aircraft'
    elif 'nabirds' in args.data_path:
        args.tag = 'nabirds'
    elif 'inat' in args.data_path:
        args.tag = 'inat'
    
    args.use_attn = attn

    
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")
    
    if os.path.exists(args.save_weights) is False:
        os.makedirs(args.save_weights)
    

    logging.basicConfig(
        filename=os.path.join(args.save_weights, args.log_name),
        filemode='w',
        format='%(asctime)s: %(message)s ',
        level=logging.INFO)
    warnings.filterwarnings("ignore")
        

    init_distributed_mode(args=args)
    rank = args.rank
    device = torch.device(args.device)
    batch_size = args.batch_size
    args.lr *= args.world_size  
    checkpoint_path = ""    
    

    set_seed(args.seed)
        
    if rank == 0:
        print('Start Tensorboard with "tensorboard --logdir={}", view at http://localhost:16006/'
          .format("runs" if args.tensorboard_dir is None else args.tensorboard_dir))
        # tensorboard init
        if args.tensorboard_dir is None:
            tb_writer = SummaryWriter()
        else:
            tb_writer = SummaryWriter(log_dir=args.tensorboard_dir)
        print(args)

    # --------------------------------------- data load -------------------------------------------- #
    train_dataset, val_dataset = get_trainval_datasets(args.tag, args.data_path, args.image_size)

    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset)
    
    train_batch_sampler = torch.utils.data.BatchSampler(
        train_sampler, batch_size, drop_last=True)
    
    train_loader = DataLoader(train_dataset,
                              batch_sampler=train_batch_sampler,
                              pin_memory=True,
                              num_workers=args.nw,
                              worker_init_fn=seed_worker)

    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            sampler=val_sampler,
                            pin_memory=True,
                            num_workers=args.nw,
                            worker_init_fn=seed_worker)

    num_classes = train_dataset.num_classes
    
    # ------------------------------------- build model --------------------------------------------- #
    model = vit_base_patch16_224_in21k(num_classes=num_classes, args=args, has_logits=False, image_size=args.image_size).to(device)
       
    if args.weights != "":
        model = load_weights(args, model)
    else:
        checkpoint_path = os.path.join(tempfile.gettempdir(), "initial_weights.pt")

        if rank == 0:
            torch.save(model.state_dict(), checkpoint_path)

        dist.barrier()
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        
    
    if args.syncBN and args.freeze_layers is False:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        
    # --------------------------------------- optimizer --------------------------------------------- #
    pg = get_params_groups(model, weight_decay=args.weights_decay)
    
    optimizer = optim.SGD(pg, lr=args.lr, momentum=args.momentum, weight_decay=args.weights_decay)
    # optimizer = optim.Adam(pg, lr=args.lr, weight_decay=args.weights_decay)
    # lr_scheduler = WarmupCosineSchedule(optimizer=optimizer,
    #                                     warmup_steps=args.warmup_epochs,
    #                                     t_total=args.epochs)
    
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs,
                                       warmup=args.warmup, warmup_epochs=args.warmup_epochs,
                                       freeze_epoch=args.freeze_epoch, freeze_lr=args.freeze_lr)
    # --------------------------------------- train -------------------------------------------- #
    if rank == 0:
        logging.info('Start training: Total epochs: {}, Batch size: {}, Training size: {}, Validation size: {}'.
                    format(args.epochs, args.batch_size, len(train_dataset), len(val_dataset)))
        logging.info('')

    best_acc = 0.
    lr_scheduler.step()
    for epoch in range(args.epochs):
        if rank == 0:
            logging.info('Epoch {:03d}, Learning Rate {:g}'.format(epoch + 1, optimizer.param_groups[0]['lr']))
        # train
        if rank == 0:
            start_time = time.time()

        train_sampler.set_epoch(epoch)

        # freeze
        if rank == 0:
            if args.freeze_epoch > 0:
                if epoch == 0:
                    for name, para in model.named_parameters():
                        if "head" not in name and "pre_logits" not in name:
                            para.requires_grad_(False)
                        else:
                            print("training {}".format(name))
                if epoch == args.freeze_epoch:
                    for name, para in model.named_parameters():
                        para.requires_grad_(True)
                    
        # train_loss, train_acc
        train_loss, train_acc = train_one_epoch(model=model,
                                                args=args,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch,
                                                lr_scheduler=lr_scheduler)
        if rank == 0:
            end_time = time.time()
            logging.info('Time: {:3.2f}'.format(end_time - start_time))
        # validate
        val_loss, val_acc = evaluate(model=model,
                                     args=args,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        if rank == 0:
            tb_writer.add_scalar(tags[0], train_loss, epoch)
            tb_writer.add_scalar(tags[1], train_acc, epoch)
            tb_writer.add_scalar(tags[2], val_loss, epoch)
            tb_writer.add_scalar(tags[3], val_acc, epoch)
            tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
        
        # lr_scheduler.step()
        
        if rank == 0:
            if best_acc < val_acc:
                torch.save(model.state_dict(), os.path.join(args.save_weights, args.save_name))
                best_acc = val_acc
            logging.info('Best Acc: {}'.format(best_acc))
            logging.info('')

    if rank == 0:
        if os.path.exists(checkpoint_path) is True:
            os.remove(checkpoint_path)

    cleanup()
    
if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    main(args)
