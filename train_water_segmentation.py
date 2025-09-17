import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.transforms.functional import to_tensor, to_pil_image
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from networks.vision_transformer import SwinUnet as ViT_seg
from config import get_config
import glob
from sklearn.metrics import jaccard_score, f1_score, accuracy_score

class WaterBodyDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, mask_dir, transform=None, img_size=224):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.img_size = img_size
        self.images = sorted(glob.glob(os.path.join(img_dir, '*.jpg')) + glob.glob(os.path.join(img_dir, '*.png')))
        self.masks = sorted(glob.glob(os.path.join(mask_dir, '*.jpg')) + glob.glob(os.path.join(mask_dir, '*.png')))
        
        # Ensure we have matching image-mask pairs
        assert len(self.images) == len(self.masks), "Number of images and masks do not match"
        for img_path, mask_path in zip(self.images, self.masks):
            img_name = os.path.basename(img_path).split('.')[0]
            mask_name = os.path.basename(mask_path).split('.')[0]
            assert img_name == mask_name, f"Mismatched image-mask pair: {img_name} vs {mask_name}"
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks[idx]
        
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # Convert to grayscale
        
        # Resize
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
        
        # Convert to tensors
        image = to_tensor(image)
        mask = to_tensor(mask)
        
        # Convert mask to binary (0 or 1)
        mask = (mask > 0.5).float()
        
        if self.transform:
            image = self.transform(image)
            
        return image, mask

def calculate_metrics(pred, target):
    """
    Calculate IoU, Dice, and Pixel Accuracy
    """
    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    
    # Flatten the arrays
    pred_flat = pred.ravel()
    target_flat = target.ravel()
    
    # Calculate metrics
    iou = jaccard_score(target_flat, pred_flat > 0.5, average='binary')
    dice = f1_score(target_flat, pred_flat > 0.5, average='binary')
    pixel_acc = accuracy_score(target_flat, pred_flat > 0.5)
    
    return iou, dice, pixel_acc

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    total_acc = 0.0
    
    for images, masks in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        iou, dice, acc = calculate_metrics(outputs, masks)
        
        running_loss += loss.item() * images.size(0)
        total_iou += iou * images.size(0)
        total_dice += dice * images.size(0)
        total_acc += acc * images.size(0)
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_iou = total_iou / len(dataloader.dataset)
    epoch_dice = total_dice / len(dataloader.dataset)
    epoch_acc = total_acc / len(dataloader.dataset)
    
    return epoch_loss, epoch_iou, epoch_dice, epoch_acc

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    total_acc = 0.0
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Calculate metrics
            iou, dice, acc = calculate_metrics(outputs, masks)
            
            running_loss += loss.item() * images.size(0)
            total_iou += iou * images.size(0)
            total_dice += dice * images.size(0)
            total_acc += acc * images.size(0)
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_iou = total_iou / len(dataloader.dataset)
    epoch_dice = total_dice / len(dataloader.dataset)
    epoch_acc = total_acc / len(dataloader.dataset)
    
    return epoch_loss, epoch_iou, epoch_dice, epoch_acc

def main():
    # Parse arguments
        # Parse arguments
    parser = argparse.ArgumentParser(description='Train Swin-UNet on Water Body Dataset')

    # Dataset / Training args
    parser.add_argument('--data_root', type=str, default='./datasets/Water Bodies Dataset',
                        help='Root directory of the dataset')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--img_size', type=int, default=224, help='Input image size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Output directory for saving models and logs')
    parser.add_argument('--pretrained_ckpt', type=str,
                        default='./pretrained_ckpt/swin_tiny_patch4_window7_224.pth',
                        help='Path to pretrained checkpoint')

    # Swin-Transformer config args
    parser.add_argument('--cfg', type=str, required=True,
                        help="path to config file")
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")
    parser.add_argument('--zip', action='store_true',
                        help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache_mode', type=str, default='part',
                        choices=['no', 'full', 'part'],
                        help='no: no cache, full: cache all data, part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', type=str, default='',
                        help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, default=0,
                        help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1',
                        choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level')
    parser.add_argument('--output', type=str, default='',
                        help='output directory')
    parser.add_argument('--tag', type=str, default='default',
                        help='tag of experiment')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true',
                        help='Test throughput only')

    args = parser.parse_args()

    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset and split into train/val
    img_dir = os.path.join(args.data_root, 'Images')
    mask_dir = os.path.join(args.data_root, 'Masks')
    
    full_dataset = WaterBodyDataset(img_dir, mask_dir, transform=transform, img_size=args.img_size)
    
    # Split dataset (80% train, 10% val, 10% test)
    train_size = int(0.8 * len(full_dataset))
    val_size = (len(full_dataset) - train_size) // 2
    test_size = len(full_dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    # Initialize model
    config = get_config(args)
    config.defrost()
    config.IMG_SIZE = args.img_size
    config.MODEL.NUM_CLASSES = 1
    config.freeze()
    
    model = ViT_seg(config, img_size=args.img_size, num_classes=1).to(device)
    
    # Load pretrained weights if available
    if os.path.exists(args.pretrained_ckpt):
        print(f"Loading pretrained weights from {args.pretrained_ckpt}")
        model.load_from(config)
    
    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5)
    
    # Training loop
    best_val_iou = 0.0
    
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        
        # Train for one epoch
        train_loss, train_iou, train_dice, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_iou, val_dice, val_acc = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f} | Train IoU: {train_iou:.4f} | "
              f"Train Dice: {train_dice:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f} | "
              f"Val Dice: {val_dice:.4f} | Val Acc: {val_acc:.4f}")
        
        # Save best model based on validation IoU
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_iou': val_iou,
                'val_dice': val_dice,
                'val_acc': val_acc,
                'args': args
            }, os.path.join(args.output_dir, 'best_model.pth'))
            print(f"Saved best model with validation IoU: {val_iou:.4f}")
            
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_iou': val_iou,
                'val_dice': val_dice,
                'val_acc': val_acc,
                'args': args
            }, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pth'))
            print(f"Saved checkpoint at epoch {epoch+1}")
    
    # Test the model
    print("\nTesting the best model...")
    checkpoint = torch.load(os.path.join(args.output_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_iou, test_dice, test_acc = validate_epoch(
        model, test_loader, criterion, device
    )
    
    print(f"\nTest Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test IoU: {test_iou:.4f}")
    print(f"Test Dice: {test_dice:.4f}")
    print(f"Test Pixel Accuracy: {test_acc:.4f}")
    
    # Save test results
    with open(os.path.join(args.output_dir, 'test_results.txt'), 'w') as f:
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test IoU: {test_iou:.4f}\n")
        f.write(f"Test Dice: {test_dice:.4f}\n")
        f.write(f"Test Pixel Accuracy: {test_acc:.4f}\n")

if __name__ == '__main__':
    main()
