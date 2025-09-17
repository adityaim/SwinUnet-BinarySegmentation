import os
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from train_water_segmentation import WaterBodyDataset, get_config
from networks.vision_transformer import SwinUnet as ViT_seg
import argparse
from tqdm import tqdm
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
import torch.nn.functional as F
from sklearn.metrics import jaccard_score, f1_score, accuracy_score

def load_model(model_path, device):
    """Load the trained model"""
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create default args if not in checkpoint
    if 'args' not in checkpoint:
        print("Warning: 'args' not found in checkpoint, using default arguments")
        class Args:
            def __init__(self):
                self.img_size = 224
                self.batch_size = 8
                self.num_workers = 4
                self.cfg = 'configs/swin_tiny_patch4_window7_224_lite.yaml'
                self.opts = []
                self.zip = False
                self.cache_mode = 'part'
                self.resume = ''
                self.accumulation_steps = 0
                self.use_checkpoint = False
                self.amp_opt_level = 'O1'
                self.output = ''
                self.tag = 'default'
                self.eval = False
                self.throughput = False
                
        args = Args()
    else:
        args = checkpoint['args']
    
    # Initialize model
    config = get_config(args)
    model = ViT_seg(config, img_size=args.img_size, num_classes=1).to(device)
    
    # Handle both state_dict formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    
    return model, args

def calculate_metrics(pred, target, threshold=0.5):
    """Calculate IoU, Dice, and Pixel Accuracy"""
    pred = (torch.sigmoid(pred) > threshold).float()
    pred_flat = pred.view(-1).cpu().numpy()
    target_flat = target.view(-1).cpu().numpy()
    
    # Calculate metrics
    iou = jaccard_score(target_flat, pred_flat, average='binary', zero_division=1)
    dice = f1_score(target_flat, pred_flat, average='binary', zero_division=1)
    acc = accuracy_score(target_flat, pred_flat)
    
    return iou, dice, acc

def evaluate_model(model, test_loader, device):
    """Evaluate model on test set and return metrics"""
    criterion = nn.BCEWithLogitsLoss()
    model.eval()
    
    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    total_acc = 0.0
    
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Calculate metrics
            iou, dice, acc = calculate_metrics(outputs, masks)
            
            total_loss += loss.item() * images.size(0)
            total_iou += iou * images.size(0)
            total_dice += dice * images.size(0)
            total_acc += acc * images.size(0)
    
    metrics = {
        'loss': total_loss / len(test_loader.dataset),
        'iou': total_iou / len(test_loader.dataset),
        'dice': total_dice / len(test_loader.dataset),
        'accuracy': total_acc / len(test_loader.dataset)
    }
    
    return metrics

def visualize_predictions(model, dataset, num_samples=5, output_dir='output/visualizations', device='cuda'):
    """Visualize model predictions on random samples"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Select random samples
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    plt.figure(figsize=(15, 5 * num_samples))
    
    for i, idx in enumerate(indices):
        image, true_mask = dataset[idx]
        image_tensor = image.unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(image_tensor)
            pred_mask = torch.sigmoid(output) > 0.5
        
        # Convert tensors to numpy arrays for visualization
        image_np = image.permute(1, 2, 0).cpu().numpy()
        true_mask_np = true_mask.squeeze().cpu().numpy()
        pred_mask_np = pred_mask.squeeze().cpu().numpy()
        
        # Denormalize image
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_np = std * image_np + mean
        image_np = np.clip(image_np, 0, 1)
        
        # Plot input image
        plt.subplot(num_samples, 3, i*3 + 1)
        plt.imshow(image_np)
        plt.title(f'Sample {i+1}: Input Image')
        plt.axis('off')
        
        # Plot ground truth mask
        plt.subplot(num_samples, 3, i*3 + 2)
        plt.imshow(true_mask_np, cmap='gray')
        plt.title('Ground Truth')
        plt.axis('off')
        
        # Plot predicted mask
        plt.subplot(num_samples, 3, i*3 + 3)
        plt.imshow(pred_mask_np, cmap='gray')
        plt.title('Prediction')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'predictions.png'), bbox_inches='tight', dpi=300)
    plt.close()
    
    # Also save individual images for better quality
    for i, idx in enumerate(indices):
        image, true_mask = dataset[idx]
        image_tensor = image.unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(image_tensor)
            pred_mask = torch.sigmoid(output) > 0.5
        
        # Convert tensors to numpy arrays
        image_np = image.permute(1, 2, 0).cpu().numpy()
        true_mask_np = true_mask.squeeze().cpu().numpy()
        pred_mask_np = pred_mask.squeeze().cpu().numpy()
        
        # Denormalize image
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_np = std * image_np + mean
        image_np = np.clip(image_np, 0, 1)
        
        # Create figure for individual sample
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        ax1.imshow(image_np)
        ax1.set_title('Input Image')
        ax1.axis('off')
        
        ax2.imshow(true_mask_np, cmap='gray')
        ax2.set_title('Ground Truth')
        ax2.axis('off')
        
        ax3.imshow(pred_mask_np, cmap='gray')
        ax3.set_title('Prediction')
        ax3.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'sample_{i+1}.png'), bbox_inches='tight', dpi=300)
        plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Water Body Segmentation Model')
    parser.add_argument('--model_path', type=str, default='output/best_model.pth',
                        help='Path to the trained model')
    parser.add_argument('--data_root', type=str, default='./datasets/Water Bodies Dataset',
                        help='Root directory of the dataset')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for evaluation')
    parser.add_argument('--img_size', type=int, default=224, help='Input image size')
    parser.add_argument('--num_samples', type=int, default=5, 
                        help='Number of samples to visualize')
    args = parser.parse_args()
    
    # Set device - force CPU since CUDA is not available
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model, model_args = load_model(args.model_path, device)
    model = model.to(device)  # Ensure model is on the correct device
    
    # Create dataset and dataloader
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = WaterBodyDataset(
        os.path.join(args.data_root, 'Images'),
        os.path.join(args.data_root, 'Masks'),
        transform=transform,
        img_size=args.img_size
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    # Evaluate model
    print("\nEvaluating model on test set...")
    metrics = evaluate_model(model, test_loader, device)
    
    print("\nTest Results:")
    print(f"Test Loss: {metrics['loss']:.4f}")
    print(f"Test IoU: {metrics['iou']:.4f}")
    print(f"Test Dice: {metrics['dice']:.4f}")
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    
    # Visualize predictions
    print("\nGenerating visualizations...")
    visualize_predictions(model, test_dataset, num_samples=min(args.num_samples, len(test_dataset)), device=device)
    print(f"Visualizations saved to output/visualizations/predictions.png")
