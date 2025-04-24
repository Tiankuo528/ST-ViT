import os
import numpy as np
import time

from tensorboardX import SummaryWriter
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


import utils

from SwinTransformer.optimizer import build_optimizer ,set_weight_decay
from SwinTransformer.lr_scheduler import LinearLRScheduler
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.step_lr import StepLRScheduler
from timm.scheduler.scheduler import Scheduler
from torch.cuda.amp import autocast, GradScaler
from models_mmst_vit_without_rotation import MMST_ViT
from dataset.temporal_loader_without_rotation import SimpleImageDataset
from timm import create_model
# try:
#     # noinspection PyUnresolvedReferences
#     from apex import amp
# except ImportError:
#     amp = None

import argparse


def parse_arguments():
    """Argument Parser for the commandline argments
    :returns: command line arguments

    """
    ##########################################################################
    #                            Training setting                            #
    ##########################################################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_classes', help="Number of classes", type=int,
                        default=2)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr_scheduler', type=str,
                        default='plateau', choices=['plateau', 'step'])
    parser.add_argument('--gamma', type=float,
                        help='LR Multiplicative factor if lr_scheduler is step',
                        default=0.1)
    parser.add_argument('--patience', type=int, default=10000)
    parser.add_argument('--log-every', type=int, default=10000)
    parser.add_argument('--save-model', type=bool, default=True)
    parser.add_argument('--backbone', type=str, default='No-organized')
    parser.add_argument('--embed_dim', default=768, type=int, help='embed dimensions') #512
    args = parser.parse_args()                     

    return args


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train_model(model, n_classes, train_loader, epoch, num_epochs, optimizer, writer, current_lr, log_every=2):
     
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()  # Set model to training mode

    scaler = GradScaler()  # Prepare the scaler for mixed precision
    
    metric = torch.nn.CrossEntropyLoss()
    losses = []
    y_probs, y_trues = np.zeros((0, n_classes), float), []

    for i, (image, label) in enumerate(train_loader):
        image, label = image.to(device), label.to(device).long()

        optimizer.zero_grad()

        # Autocast context for mixed precision
        with autocast():
            prediction = model(image)
            loss = metric(prediction, label)
        
        # Backpropagation
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_value = loss.item()
        losses.append(loss_value)
        
        # Probability calculations for accuracy metrics, not suitable under autocast
        y_prob = F.softmax(prediction.detach().cpu().float(), dim=1)
        y_probs = np.concatenate([y_probs, y_prob.numpy()])
        y_trues.extend(label.cpu().tolist())

        # Logging
        if i % log_every == 0:
            metric_collects = utils.calc_multi_cls_measures(y_probs, y_trues)
            writer.add_scalar('Train/Loss', loss_value, epoch * len(train_loader) + i)
            writer.add_scalar('Train/ACC', metric_collects['accuracy'], epoch * len(train_loader) + i)
            utils.print_progress(epoch + 1, num_epochs, i, len(train_loader), np.mean(losses), current_lr, metric_collects)

    train_loss_epoch = np.mean(losses)
    return train_loss_epoch, utils.calc_multi_cls_measures(y_probs, y_trues)

def evaluate_model(model,n_classes, val_loader, epoch, num_epochs, writer, current_lr,
                   log_every=4000):
    
    metric = torch.nn.CrossEntropyLoss()

    model.eval()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.train()
            m.weight.requires_grad = False
            m.bias.requires_grad = False

    y_probs = np.zeros((0, n_classes), float)
    losses, y_trues = [], []

    for i, (image, label) in enumerate(val_loader):

        if torch.cuda.is_available():
            image = image.cuda()
            label = label.cuda()

        prediction = model(image.float().cuda())
        loss = metric(prediction, label.long())

        loss_value = loss.item()
        losses.append(loss_value)
        y_prob = F.softmax(prediction, dim=1)
        y_probs = np.concatenate([y_probs, y_prob.detach().cpu().numpy()])
        y_trues.append(label.item())

        metric_collects = utils.calc_multi_cls_measures(y_probs, y_trues)

        n_iter = epoch * len(val_loader) + i
        writer.add_scalar('Val/Loss', loss_value, n_iter)
        writer.add_scalar('Val/ACC', metric_collects['accuracy'], n_iter)

        if (i % log_every == 0) & (i > 0):
            prefix = '*Val|'
            utils.print_progress(epoch + 1, num_epochs, i, len(val_loader),
                                 np.mean(losses), current_lr, metric_collects,
                                 prefix=prefix)

    val_loss_epoch = np.round(np.mean(losses), 4)
    return val_loss_epoch, metric_collects


def main(args):
    """Main function for the training pipeline

    :args: commandlien arguments
    :returns: None

    """
    ##########################################################################
    #                             Basic settings                             #
    ##########################################################################
    exp_dir = 'experiments'+args.backbone
    log_dir = os.path.join(exp_dir, 'logs')
    model_dir = os.path.join(exp_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)

    ##########################################################################
    #  Define all the necessary variables for model training and evaluation  #
    #../../dataset/MIA-COV19-DATA/data/  #data sample  for debuging
    #../../dataset/MIA-COV19-DATA/Data/Lung/ICCV_Lung_split/data/
    ##########################################################################
    writer = SummaryWriter(log_dir)
    train_dataset = SimpleImageDataset('/mnt/e/Tiankuo/X-ray dataset', is_train=True)  



    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, num_workers=5,
        drop_last=False)

    val_dataset = SimpleImageDataset('/mnt/e/Tiankuo/X-ray dataset', is_train=False) 
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=5,
        drop_last=False)



    swin_model = create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=0)
    
    # Initialize the MMST_ViT with the Swin Transformer backbone
    model = MMST_ViT(out_dim=2, swin_model=swin_model, dim=args.embed_dim) #batch_size=args.batch_size

    if torch.cuda.is_available():
        model = model.cuda()
    ######################################################
    # thansformer optimizer
    ######################################################
    """
        Build optimizer, set weight decay of normalization to 0 by default.
        """
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
    parameters = set_weight_decay(model, skip, skip_keywords)

    optimizer = optim.AdamW(parameters, eps=1.0e-08, betas=(0.9, 0.999),
                            lr=1e-5, weight_decay=0.05)  #7.8425e-05 weight decay =0.05
    # transkformer optimizer#########################################################
    #lr_scheduler = build_scheduler(config, optimizer, len(train_loader))
    #swin_model, optimizer = amp.initialize(swin_model, optimizer, opt_level='O1')
    if args.lr_scheduler == "plateau":
            scheduler = LinearLRScheduler(optimizer, t_initial=50*len(train_loader), lr_min_rate=0.01, warmup_lr_init=5e-7,
            warmup_t=1*len(train_loader), t_in_epochs=False,)

        # scheduler = CosineLRScheduler(
        #     optimizer,
        #     t_initial=30*len(train_loader),
        #     t_mul=1.,
        #     lr_min=0.01,
        #     warmup_lr_init=5e-7,
        #     warmup_t=5*len(train_loader),
        #     cycle_limit=1,
        #     t_in_epochs=False,
        # )
    elif args.lr_scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=3, gamma=args.gamma)
    

    best_val_loss = float('inf')
    best_val_accu = float(0)

    iteration_change_loss = 0
    t_start_training = time.time()
    ##########################################################################
    #                           Main training loop                           #
    ##########################################################################
    num_steps = len(train_loader)
    for epoch in range(args.epochs):
        current_lr = get_lr(optimizer)
        t_start = time.time()

        ############################################################
        #  The actual training and validation step for each epoch  #
        ############################################################
        n_classes = 2  # Define the number of classes appropriately based on your use case
        train_loss, train_metric = train_model(
            model, n_classes, train_loader, epoch, args.epochs, optimizer, writer, args.lr, args.log_every)


        with torch.no_grad():
            val_loss, val_metric = evaluate_model(
                model, n_classes,val_loader, epoch, args.epochs, writer, current_lr)

        ##############################
        #  Adjust the learning rate  #
        ##############################
        if args.lr_scheduler == 'plateau':
            if (args.backbone == 'COV') | (args.backbone == 'BiT') | (args.backbone == 'Effv2'):
                scheduler.step(val_loss)
            else:
                scheduler.step_update(epoch * num_steps)
        elif args.lr_scheduler == 'step':
            scheduler.step()

        t_end = time.time()
        delta = t_end - t_start

        utils.print_epoch_progress(train_loss, val_loss, delta, train_metric,
                                   val_metric)
        iteration_change_loss += 1
        if (val_metric['precisions'][1:] != 0) & (val_metric['recalls'][1:] != 0):
            F1 = (2 * val_metric['precision'
                                 's'][1:] * val_metric['recalls'][1:]) / (val_metric['precisions'][1:] + val_metric['recalls'][1:])
        else:
            F1= 'NAN'
        print('-' * 30+'F1 = '+str(F1)+'-' * 30)

        train_acc, val_acc = train_metric['accuracy'], val_metric['accuracy']
        file_name = ('train_acc_{}_val_acc_{}_epoch_{}.pth'.
                     format(train_acc, val_acc, epoch))
        torch.save(model, os.path.join(model_dir, file_name))

        if val_acc > best_val_accu:
            best_val_accu = val_acc
            if bool(args.save_model):
                torch.save(model, os.path.join(model_dir, 'best.pth'))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            iteration_change_loss = 0

        if iteration_change_loss == args.patience:
            print(('Early stopping after {0} iterations without the decrease ' +
                  'of the val loss').format(iteration_change_loss))
            break
    t_end_training = time.time()
    print('training took {}s'.
          format(t_end_training - t_start_training))


if __name__ == "__main__":
    args = parse_arguments()
    main(args)