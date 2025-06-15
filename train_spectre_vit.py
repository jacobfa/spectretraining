#!/usr/bin/env python3
"""
Training script for SPECTRE Vision Transformer on ImageNet

This script trains the SPECTRE ViT model with:
- Distributed Data Parallel (DDP) support
- Comprehensive logging and metrics tracking
- Model checkpointing and best model saving
- Progress tracking with tqdm
- Validation and testing
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
from datetime import datetime
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageNet

from tqdm import tqdm
import numpy as np

# Import SPECTRE model
from spectre import create_spectre_vit


# Advanced data augmentation utilities
class RandAugment:
    """RandAugment implementation for ViT training."""
    
    def __init__(self, n=2, m=9):
        self.n = n  # Number of augmentation transformations to apply
        self.m = m  # Magnitude for all the transformations
        
        # Define augmentation operations
        self.augment_list = [
            (self.auto_contrast, 0, 1),
            (self.equalize, 0, 1),
            (self.invert, 0, 1),
            (self.rotate, 0, 30),
            (self.posterize, 0, 4),
            (self.solarize, 0, 256),
            (self.color, 0.1, 1.9),
            (self.contrast, 0.1, 1.9),
            (self.brightness, 0.1, 1.9),
            (self.sharpness, 0.1, 1.9),
            (self.cutout, 0, 0.2),
        ]
    
    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op, minval, maxval in ops:
            val = (float(self.m) / 30) * float(maxval - minval) + minval
            img = op(img, val)
        return img
    
    def auto_contrast(self, pil_img, level):
        return transforms.functional.autocontrast(pil_img)
    
    def equalize(self, pil_img, level):
        return transforms.functional.equalize(pil_img)
    
    def invert(self, pil_img, level):
        return transforms.functional.invert(pil_img)
    
    def rotate(self, pil_img, level):
        degrees = int(level)
        if random.random() < 0.5:
            degrees = -degrees
        return transforms.functional.rotate(pil_img, degrees)
    
    def posterize(self, pil_img, level):
        bits = int(level)
        return transforms.functional.posterize(pil_img, bits)
    
    def solarize(self, pil_img, level):
        threshold = int(level)
        return transforms.functional.solarize(pil_img, threshold)
    
    def color(self, pil_img, level):
        return transforms.functional.adjust_saturation(pil_img, level)
    
    def contrast(self, pil_img, level):
        return transforms.functional.adjust_contrast(pil_img, level)
    
    def brightness(self, pil_img, level):
        return transforms.functional.adjust_brightness(pil_img, level)
    
    def sharpness(self, pil_img, level):
        return transforms.functional.adjust_sharpness(pil_img, level)
    
    def cutout(self, pil_img, level):
        # Convert to tensor for cutout
        tensor_img = transforms.functional.to_tensor(pil_img)
        h, w = tensor_img.shape[1], tensor_img.shape[2]
        cutout_size = int(level * min(h, w))
        
        if cutout_size > 0:
            mask = torch.ones(h, w)
            y = random.randint(0, h - cutout_size) if h > cutout_size else 0
            x = random.randint(0, w - cutout_size) if w > cutout_size else 0
            mask[y:y+cutout_size, x:x+cutout_size] = 0
            tensor_img = tensor_img * mask.unsqueeze(0)
        
        return transforms.functional.to_pil_image(tensor_img)


class MixUp:
    """MixUp augmentation for ViT training."""
    
    def __init__(self, alpha=0.2):
        self.alpha = alpha
    
    def __call__(self, batch, targets):
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = batch.size(0)
        index = torch.randperm(batch_size).to(batch.device)
        
        mixed_batch = lam * batch + (1 - lam) * batch[index, ...]
        targets_a, targets_b = targets, targets[index]
        
        return mixed_batch, targets_a, targets_b, lam


class CutMix:
    """CutMix augmentation for ViT training."""
    
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def __call__(self, batch, targets):
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = batch.size(0)
        index = torch.randperm(batch_size).to(batch.device)
        
        # Generate random bounding box
        W, H = batch.size(3), batch.size(2)
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        batch[:, :, bby1:bby2, bbx1:bbx2] = batch[index, :, bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda to match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        
        targets_a, targets_b = targets, targets[index]
        return batch, targets_a, targets_b, lam


class ExponentialMovingAverage:
    """Exponential Moving Average for model parameters."""
    
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixed loss function for MixUp/CutMix."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def setup_logging(log_dir: Path, rank: int) -> logging.Logger:
    """Setup logging configuration."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(f'spectre_training_rank_{rank}')
    logger.setLevel(logging.INFO)
    
    # Create handlers
    log_file = log_dir / f'training_rank_{rank}.log'
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - Rank %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    if rank == 0:  # Only rank 0 logs to console
        logger.addHandler(console_handler)
    
    return logger


def get_data_loaders(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    world_size: int,
    rank: int,
    img_size: int = 224,
    use_advanced_aug: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """Create ImageNet data loaders with advanced augmentations for ViT training."""
    
    if use_advanced_aug:
        # Advanced augmentation pipeline for maximum performance
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.08, 1.0), ratio=(3./4., 4./3.)),
            transforms.RandomHorizontalFlip(p=0.5),
            RandAugment(n=2, m=9),  # RandAugment for better generalization
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3)),  # Random erasing
        ])
    else:
        # Standard augmentation pipeline
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    # Enhanced validation transform with multiple scales
    val_transform = transforms.Compose([
        transforms.Resize(int(img_size * 1.143)),  # 256 for 224
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = ImageNet(
        root=data_dir,
        split='train',
        transform=train_transform
    )
    
    val_dataset = ImageNet(
        root=data_dir,
        split='val',
        transform=val_transform
    )
    
    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True  # Drop last for consistent batch sizes
    )
    
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    
    # Create data loaders with optimized settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    return train_loader, val_loader


class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)) -> list:
    """Computes the accuracy over the k top predictions."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        
        return res


def save_checkpoint(
    state: Dict[str, Any],
    is_best: bool,
    checkpoint_dir: Path,
    filename: str = 'checkpoint.pth'
):
    """Save model checkpoint."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    filepath = checkpoint_dir / filename
    torch.save(state, filepath)
    
    if is_best:
        best_filepath = checkpoint_dir / 'best_model.pth'
        torch.save(state, best_filepath)


def load_checkpoint(filepath: Path, model: nn.Module, optimizer: optim.Optimizer = None) -> Dict:
    """Load model checkpoint."""
    if not filepath.exists():
        return {'epoch': 0, 'best_acc1': 0}
    
    checkpoint = torch.load(filepath, map_location='cpu')
    
    # Handle DDP state dict mismatch
    state_dict = checkpoint['state_dict']
    model_state_dict = model.state_dict()
    
    # Check if there's a mismatch between DDP and non-DDP models
    is_model_ddp = hasattr(model, 'module')
    is_checkpoint_ddp = any(key.startswith('module.') for key in state_dict.keys())
    
    if is_model_ddp and not is_checkpoint_ddp:
        # Loading non-DDP checkpoint into DDP model
        # Add 'module.' prefix to checkpoint keys
        state_dict = {f'module.{key}': value for key, value in state_dict.items()}
    elif not is_model_ddp and is_checkpoint_ddp:
        # Loading DDP checkpoint into non-DDP model
        # Remove 'module.' prefix from checkpoint keys
        state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
    
    # Load the adjusted state dict
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print(f"Warning: Failed to load checkpoint with error: {e}")
        print("Attempting to load with strict=False...")
        model.load_state_dict(state_dict, strict=False)
    
    if optimizer and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    return checkpoint


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    device: torch.device,
    logger: logging.Logger,
    writer: SummaryWriter = None,
    rank: int = 0,
    mixup: MixUp = None,
    cutmix: CutMix = None,
    ema: ExponentialMovingAverage = None,
    clip_grad: float = 1.0,
    mixup_prob: float = 0.5
) -> Dict[str, float]:
    """Train for one epoch with advanced techniques."""
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    model.train()
    
    if rank == 0:
        pbar = tqdm(
            train_loader,
            desc=f'Train Epoch {epoch}',
            unit='batch',
            dynamic_ncols=True
        )
    else:
        pbar = train_loader
    
    end = time.time()
    
    for i, (images, targets) in enumerate(pbar):
        # Measure data loading time
        data_time.update(time.time() - end)
        
        # Move data to device
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        # Apply MixUp or CutMix with probability
        use_mixup = mixup is not None and random.random() < mixup_prob
        use_cutmix = cutmix is not None and random.random() < mixup_prob and not use_mixup
        
        if use_mixup:
            images, targets_a, targets_b, lam = mixup(images, targets)
        elif use_cutmix:
            images, targets_a, targets_b, lam = cutmix(images, targets)
        
        # Forward pass
        outputs = model(images)
        
        # Calculate loss
        if use_mixup or use_cutmix:
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            # For accuracy calculation, use the original targets (approximation)
            acc_targets = targets_a if lam > 0.5 else targets_b
        else:
            loss = criterion(outputs, targets)
            acc_targets = targets
        
        # Measure accuracy
        acc1, acc5 = accuracy(outputs, acc_targets, topk=(1, 5))
        
        # Update meters
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))
        top5.update(acc5[0].item(), images.size(0))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        
        optimizer.step()
        
        # Update EMA
        if ema is not None:
            ema.update()
        
        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        # Update progress bar
        if rank == 0:
            postfix = {
                'Loss': f'{losses.val:.4f} ({losses.avg:.4f})',
                'Acc@1': f'{top1.val:.2f} ({top1.avg:.2f})',
                'Acc@5': f'{top5.val:.2f} ({top5.avg:.2f})',
                'Time': f'{batch_time.val:.3f}s'
            }
            
            if use_mixup:
                postfix['Aug'] = 'MixUp'
            elif use_cutmix:
                postfix['Aug'] = 'CutMix'
            
            pbar.set_postfix(postfix)
        
        # Log to tensorboard more frequently
        if writer and rank == 0 and i % 50 == 0:
            global_step = epoch * len(train_loader) + i
            writer.add_scalar('Train/Loss_Step', losses.val, global_step)
            writer.add_scalar('Train/Acc@1_Step', top1.val, global_step)
            writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], global_step)
    
    # Log epoch results
    if rank == 0:
        logger.info(
            f'Train Epoch {epoch}: '
            f'Loss {losses.avg:.4f}, '
            f'Acc@1 {top1.avg:.2f}%, '
            f'Acc@5 {top5.avg:.2f}%, '
            f'Time {batch_time.sum:.1f}s, '
            f'Data {data_time.sum:.1f}s'
        )
    
    return {
        'loss': losses.avg,
        'acc1': top1.avg,
        'acc5': top5.avg,
        'time': batch_time.sum,
        'data_time': data_time.sum
    }


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    logger: logging.Logger,
    epoch: int = None,
    rank: int = 0
) -> Dict[str, float]:
    """Validate the model."""
    
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    model.eval()
    
    if rank == 0:
        pbar = tqdm(
            val_loader,
            desc=f'Val Epoch {epoch}' if epoch else 'Validation',
            unit='batch',
            dynamic_ncols=True
        )
    else:
        pbar = val_loader
    
    with torch.no_grad():
        end = time.time()
        
        for i, (images, targets) in enumerate(pbar):
            # Move data to device
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            # Measure accuracy
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            
            # Update meters
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0].item(), images.size(0))
            top5.update(acc5[0].item(), images.size(0))
            
            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            # Update progress bar
            if rank == 0:
                pbar.set_postfix({
                    'Loss': f'{losses.val:.4f} ({losses.avg:.4f})',
                    'Acc@1': f'{top1.val:.2f} ({top1.avg:.2f})',
                    'Acc@5': f'{top5.val:.2f} ({top5.avg:.2f})'
                })
    
    # Log validation results
    if rank == 0:
        logger.info(
            f'Validation: '
            f'Loss {losses.avg:.4f}, '
            f'Acc@1 {top1.avg:.2f}%, '
            f'Acc@5 {top5.avg:.2f}%'
        )
    
    return {
        'loss': losses.avg,
        'acc1': top1.avg,
        'acc5': top5.avg
    }


def main():
    parser = argparse.ArgumentParser(description='Train SPECTRE ViT on ImageNet with Advanced Techniques')
    
    # Data arguments
    parser.add_argument('--data-dir', default='/data/jacob/ImageNet/', 
                        help='Path to ImageNet dataset')
    parser.add_argument('--output-dir', default='./runs', 
                        help='Output directory for logs and checkpoints')
    
    # Model arguments
    parser.add_argument('--img-size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--patch-size', type=int, default=16,
                        help='Patch size')
    parser.add_argument('--embed-dim', type=int, default=768,
                        help='Embedding dimension')
    parser.add_argument('--depth', type=int, default=12,
                        help='Number of transformer blocks')
    parser.add_argument('--n-heads', type=int, default=12,
                        help='Number of attention heads')
    parser.add_argument('--mlp-ratio', type=float, default=4.0,
                        help='MLP expansion ratio')
    parser.add_argument('--use-wavelet', action='store_true',
                        help='Use wavelet refinement module')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size per GPU')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='Weight decay')
    parser.add_argument('--warmup-epochs', type=int, default=20,
                        help='Warmup epochs')
    
    # Advanced training arguments
    parser.add_argument('--use-advanced-aug', action='store_true', default=True,
                        help='Use advanced data augmentation')
    parser.add_argument('--mixup-alpha', type=float, default=0.2,
                        help='MixUp alpha parameter')
    parser.add_argument('--cutmix-alpha', type=float, default=1.0,
                        help='CutMix alpha parameter')
    parser.add_argument('--mixup-prob', type=float, default=0.5,
                        help='Probability of applying MixUp/CutMix')
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                        help='Label smoothing epsilon')
    parser.add_argument('--clip-grad', type=float, default=1.0,
                        help='Gradient clipping norm')
    parser.add_argument('--use-ema', action='store_true', default=True,
                        help='Use Exponential Moving Average')
    parser.add_argument('--ema-decay', type=float, default=0.9999,
                        help='EMA decay rate')
    
    # Scheduler arguments
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'step', 'exponential'],
                        help='Learning rate scheduler')
    parser.add_argument('--step-size', type=int, default=30,
                        help='Step size for step scheduler')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='Gamma for step/exponential scheduler')
    
    # System arguments
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Number of data loading workers')
    parser.add_argument('--resume', type=str, default='',
                        help='Path to checkpoint to resume from')
    
    # DDP arguments
    parser.add_argument('--local-rank', type=int, default=0,
                        help='Local rank for distributed training')
    
    args = parser.parse_args()
    
    # Initialize distributed training
    if 'WORLD_SIZE' in os.environ:
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.rank = int(os.environ['RANK'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
    else:
        args.world_size = 1
        args.rank = 0
        args.local_rank = 0
    
    # Initialize process group
    if args.world_size > 1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)
    
    device = torch.device(f'cuda:{args.local_rank}')
    
    # Set random seeds for reproducibility
    torch.manual_seed(42 + args.rank)
    np.random.seed(42 + args.rank)
    random.seed(42 + args.rank)
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f'spectre_vit_advanced_{timestamp}'
    output_dir = Path(args.output_dir) / run_name
    
    # Setup logging
    logger = setup_logging(output_dir / 'logs', args.rank)
    
    if args.rank == 0:
        logger.info(f'Starting SPECTRE ViT training with advanced techniques')
        logger.info(f'Arguments: {vars(args)}')
        
        # Save arguments
        with open(output_dir / 'args.json', 'w') as f:
            json.dump(vars(args), f, indent=2)
        
        # Setup tensorboard
        writer = SummaryWriter(output_dir / 'tensorboard')
    else:
        writer = None
    
    # Create model
    model = create_spectre_vit(
        img_size=args.img_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        n_heads=args.n_heads,
        mlp_ratio=args.mlp_ratio,
        num_classes=1000,
        use_wavelet=args.use_wavelet
    )
    
    model = model.to(device)
    
    if args.rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f'Total parameters: {total_params:,}')
        logger.info(f'Trainable parameters: {trainable_params:,}')
    
    # Initialize EMA before DDP wrapping
    ema = None
    if args.use_ema:
        ema = ExponentialMovingAverage(model, decay=args.ema_decay)
        if args.rank == 0:
            logger.info(f'Using EMA with decay {args.ema_decay}')
    
    # Wrap model with DDP
    if args.world_size > 1:
        model = DDP(model, device_ids=[args.local_rank])
    
    # Create optimizer with improved settings
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        eps=1e-8,
        betas=(0.9, 0.999)
    )
    
    # Create advanced scheduler
    if args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=1e-6
        )
    elif args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.step_size,
            gamma=args.gamma
        )
    else:  # exponential
        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=args.gamma
        )
    
    # Warmup scheduler
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        total_iters=args.warmup_epochs
    )
    
    # Initialize augmentation techniques
    mixup = MixUp(alpha=args.mixup_alpha) if args.mixup_alpha > 0 else None
    cutmix = CutMix(alpha=args.cutmix_alpha) if args.cutmix_alpha > 0 else None
    
    if args.rank == 0:
        if mixup:
            logger.info(f'Using MixUp with alpha {args.mixup_alpha}')
        if cutmix:
            logger.info(f'Using CutMix with alpha {args.cutmix_alpha}')
    
    # Load checkpoint if resuming
    start_epoch = 0
    best_acc1 = 0
    
    if args.resume:
        checkpoint_path = Path(args.resume)
        checkpoint = load_checkpoint(checkpoint_path, model, optimizer)
        start_epoch = checkpoint.get('epoch', 0)
        best_acc1 = checkpoint.get('best_acc1', 0)
        
        # Restore EMA if available
        if ema and 'ema_state_dict' in checkpoint:
            ema.shadow = checkpoint['ema_state_dict']
        
        if args.rank == 0:
            logger.info(f'Resumed from epoch {start_epoch}, best acc1: {best_acc1:.2f}%')
    
    # Create data loaders with advanced augmentation
    train_loader, val_loader = get_data_loaders(
        args.data_dir,
        args.batch_size,
        args.num_workers,
        args.world_size,
        args.rank,
        args.img_size,
        use_advanced_aug=args.use_advanced_aug
    )
    
    # Loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    
    # Training stats
    train_stats = []
    val_stats = []
    
    if args.rank == 0:
        logger.info('Starting training with advanced techniques...')
        logger.info(f'Advanced augmentation: {args.use_advanced_aug}')
        logger.info(f'Label smoothing: {args.label_smoothing}')
        logger.info(f'Gradient clipping: {args.clip_grad}')
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        # Set epoch for distributed sampler
        if args.world_size > 1:
            train_loader.sampler.set_epoch(epoch)
        
        # Train for one epoch with advanced techniques
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, epoch,
            device, logger, writer, args.rank,
            mixup=mixup, cutmix=cutmix, ema=ema,
            clip_grad=args.clip_grad, mixup_prob=args.mixup_prob
        )
        
        # Update learning rate
        if epoch < args.warmup_epochs:
            warmup_scheduler.step()
        else:
            scheduler.step()
        
        # Validate with EMA if available
        if ema and args.rank == 0:
            ema.apply_shadow()
        
        val_metrics = validate(
            model, val_loader, criterion, device, logger, epoch, args.rank
        )
        
        if ema and args.rank == 0:
            ema.restore()
        
        # Check if this is the best model
        is_best = val_metrics['acc1'] > best_acc1
        best_acc1 = max(val_metrics['acc1'], best_acc1)
        
        # Save checkpoint
        if args.rank == 0:
            # Add epoch info to metrics
            train_metrics['epoch'] = epoch
            train_metrics['lr'] = optimizer.param_groups[0]['lr']
            val_metrics['epoch'] = epoch
            val_metrics['is_best'] = is_best
            
            train_stats.append(train_metrics)
            val_stats.append(val_metrics)
            
            # Prepare checkpoint state
            checkpoint_state = {
                'epoch': epoch + 1,
                'state_dict': model.module.state_dict() if args.world_size > 1 else model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_acc1': best_acc1,
                'args': vars(args)
            }
            
            # Add EMA state if available
            if ema:
                checkpoint_state['ema_state_dict'] = ema.shadow
            
            save_checkpoint(
                checkpoint_state,
                is_best,
                output_dir / 'checkpoints',
                f'checkpoint_epoch_{epoch:03d}.pth'
            )
            
            # Save latest checkpoint
            save_checkpoint(
                checkpoint_state,
                False,
                output_dir / 'checkpoints',
                'latest.pth'
            )
            
            # Save training stats
            with open(output_dir / 'train_stats.json', 'w') as f:
                json.dump(train_stats, f, indent=2)
            
            with open(output_dir / 'val_stats.json', 'w') as f:
                json.dump(val_stats, f, indent=2)
            
            # Log to tensorboard
            if writer:
                writer.add_scalar('Epoch/Train_Loss', train_metrics['loss'], epoch)
                writer.add_scalar('Epoch/Train_Acc1', train_metrics['acc1'], epoch)
                writer.add_scalar('Epoch/Val_Loss', val_metrics['loss'], epoch)
                writer.add_scalar('Epoch/Val_Acc1', val_metrics['acc1'], epoch)
                writer.add_scalar('Epoch/Learning_Rate', train_metrics['lr'], epoch)
                writer.add_scalar('Epoch/Best_Acc1', best_acc1, epoch)
            
            logger.info(
                f'Epoch {epoch}: '
                f'Train Loss {train_metrics["loss"]:.4f}, '
                f'Train Acc@1 {train_metrics["acc1"]:.2f}%, '
                f'Val Loss {val_metrics["loss"]:.4f}, '
                f'Val Acc@1 {val_metrics["acc1"]:.2f}%, '
                f'Best Acc@1 {best_acc1:.2f}%, '
                f'LR {train_metrics["lr"]:.2e}'
            )
    
    # Final validation with EMA
    if ema and args.rank == 0:
        logger.info('Final validation with EMA...')
        ema.apply_shadow()
        final_val_metrics = validate(
            model, val_loader, criterion, device, logger, args.epochs, args.rank
        )
        logger.info(f'Final EMA validation accuracy: {final_val_metrics["acc1"]:.2f}%')
    
    # Cleanup
    if args.world_size > 1:
        dist.destroy_process_group()
    
    if args.rank == 0:
        logger.info(f'Training completed! Best validation accuracy: {best_acc1:.2f}%')
        if writer:
            writer.close()


if __name__ == '__main__':
    main() 
