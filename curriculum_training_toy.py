'''
python script_name.py --batch-size 64 --learning-rate 0.0005 --epochs-per-stage 3

python script_name.py --resolutions 8 16 24 32 --epochs-per-stage 2

python curriculum_training_toy.py --epochs-per-stage 5 --checkpoint-interval 2

python curriculum_training_toy.py --resume-curriculum ./checkpoints/curriculum_res24_checkpoint_epoch_2.pth.tar --resume-regular ./checkpoints/regular_checkpoint_epoch_4.pth.tar

python curriculum_training_toy.py --no-cuda --batch-size 16

python curriculum_training_toy.py --seed 42

python curriculum_training_toy.py --num-workers 4 --batch-size 128

python curriculum_training_toy.py \
  --batch-size 64 \
  --learning-rate 0.001 \
  --epochs-per-stage 3 \
  --resolutions 16 24 32 48 \
  --final-resolution 48 \
  --accuracy-threshold 0.6 \
  --seed 123 \
  --num-workers 4 \
  --checkpoint-dir ./toy_checkpoints \
  --output-dir ./toy_results \
  --checkpoint-interval 1

'''


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
# from scipy.integrate import trapz
from scipy.integrate import trapezoid as trapz

import os
import json
import logging
import argparse
import csv
import time
from datetime import datetime
import sys
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from IPython.display import display

# Set up logging
def setup_logging(log_dir="./logs"):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

# Define a simple CNN with global average pooling to handle different input sizes
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.global_pool(x)
        x = x.view(-1, 128)
        x = self.fc(x)
        return x

# Function to get DataLoader for a specific resolution
def get_dataloader(resolution, batch_size=32, train=True, num_workers=2):
    try:
        transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = torchvision.datasets.CIFAR10(root='./data', train=train, download=True, transform=transform)
        return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=num_workers, pin_memory=True)
    except Exception as e:
        logger.error(f"Error creating dataloader: {str(e)}")
        raise

# Save checkpoint function
def save_checkpoint(state, is_best=False, checkpoint_dir='./checkpoints', filename='checkpoint.pth.tar'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    logger.info(f"Checkpoint saved to {filepath}")
    if is_best:
        best_filepath = os.path.join(checkpoint_dir, 'model_best.pth.tar')
        torch.save(state, best_filepath)
        logger.info(f"Best model saved to {best_filepath}")

# Load checkpoint function
def load_checkpoint(checkpoint_path, model, optimizer=None):
    if not os.path.exists(checkpoint_path):
        logger.warning(f"No checkpoint found at {checkpoint_path}")
        return None, 0, {}

    try:
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        start_epoch = checkpoint['epoch']
        history = checkpoint.get('history', {'loss': [], 'accuracy': []})
        config = checkpoint.get('config', {})

        logger.info(f"Loaded checkpoint from epoch {start_epoch}")
        return model, start_epoch, history, config
    except Exception as e:
        logger.error(f"Error loading checkpoint: {str(e)}")
        return None, 0, {}, {}

# Save history to CSV
def save_history_to_csv(history, filename='training_history.csv'):
    try:
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['epoch'] + list(history.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for i in range(len(history['loss'])):
                row = {'epoch': i+1}
                for key in history.keys():
                    row[key] = history[key][i]
                writer.writerow(row)
        logger.info(f"Training history saved to {filename}")
    except Exception as e:
        logger.error(f"Error saving history to CSV: {str(e)}")

# Training function that returns loss and accuracy history
def train_model(model, dataloader, optimizer, criterion, num_epochs, device,
                start_epoch=0, history=None, checkpoint_dir='./checkpoints',
                checkpoint_interval=1, model_name='model'):
    if history is None:
        history = {'loss': [], 'accuracy': []}

    model.to(device)
    best_acc = max(history['accuracy']) if history['accuracy'] else 0

    for epoch in range(start_epoch, start_epoch + num_epochs):
        try:
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            # Training loop
            for batch_idx, (images, labels) in enumerate(dataloader):
                try:
                    images, labels = images.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    # Print batch progress
                    if (batch_idx + 1) % 50 == 0:
                        logger.info(f"Epoch [{epoch+1}/{start_epoch+num_epochs}], "
                                    f"Batch [{batch_idx+1}/{len(dataloader)}], "
                                    f"Loss: {loss.item():.4f}")
                except Exception as e:
                    logger.error(f"Error in training batch {batch_idx}: {str(e)}")
                    continue

            # Calculate epoch metrics
            epoch_loss = running_loss / len(dataloader)
            epoch_acc = correct / total

            # Update history
            history['loss'].append(epoch_loss)
            history['accuracy'].append(epoch_acc)

            logger.info(f"Epoch [{epoch+1}/{start_epoch+num_epochs}], "
                        f"Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

            # Save checkpoint
            is_best = epoch_acc > best_acc
            if is_best:
                best_acc = epoch_acc

            if (epoch + 1) % checkpoint_interval == 0 or is_best:
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': epoch_loss,
                    'accuracy': epoch_acc,
                    'history': history,
                    'config': {
                        'model_name': model_name,
                        'epochs': num_epochs,
                        'start_epoch': start_epoch
                    }
                }

                save_checkpoint(
                    checkpoint,
                    is_best=is_best,
                    checkpoint_dir=checkpoint_dir,
                    filename=f'{model_name}_checkpoint_epoch_{epoch+1}.pth.tar'
                )

                # Save current history to CSV
                save_history_to_csv(history, f'{checkpoint_dir}/{model_name}_history.csv')

        except KeyboardInterrupt:
            logger.info("Training interrupted by user. Saving checkpoint...")
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss if 'epoch_loss' in locals() else 0,
                'accuracy': epoch_acc if 'epoch_acc' in locals() else 0,
                'history': history,
                'config': {
                    'model_name': model_name,
                    'epochs': num_epochs,
                    'start_epoch': start_epoch
                }
            }
            save_checkpoint(
                checkpoint,
                checkpoint_dir=checkpoint_dir,
                filename=f'{model_name}_interrupted_epoch_{epoch+1}.pth.tar'
            )
            save_history_to_csv(history, f'{checkpoint_dir}/{model_name}_history_interrupted.csv')
            raise
        except Exception as e:
            logger.error(f"Error in epoch {epoch+1}: {str(e)}")
            # Save checkpoint on error
            if 'epoch_loss' in locals() and 'epoch_acc' in locals():
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': epoch_loss,
                    'accuracy': epoch_acc,
                    'history': history,
                    'config': {
                        'model_name': model_name,
                        'epochs': num_epochs,
                        'start_epoch': start_epoch
                    }
                }
                save_checkpoint(
                    checkpoint,
                    checkpoint_dir=checkpoint_dir,
                    filename=f'{model_name}_error_epoch_{epoch+1}.pth.tar'
                )
                save_history_to_csv(history, f'{checkpoint_dir}/{model_name}_history_error.csv')
            continue

    return history

# Evaluation function
def evaluate_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    try:
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        return accuracy
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        return 0.0

# Function to find the first epoch where accuracy exceeds a threshold
def first_epoch_above_threshold(history, threshold):
    for epoch, acc in enumerate(history['accuracy']):
        if acc > threshold:
            return epoch + 1
    return None


# Corrected plot_training_curves function
def plot_training_curves(curriculum_history, regular_history, total_epochs, save_dir='./plots'):
    os.makedirs(save_dir, exist_ok=True)
    epochs = list(range(1, total_epochs + 1))  # Convert range to list for Plotly

    try:
        # Create a DataFrame for easier data handling
        df = pd.DataFrame({
            'Epoch': epochs,
            'Curriculum Loss': curriculum_history['loss'],
            'Regular Loss': regular_history['loss'],
            'Curriculum Accuracy': curriculum_history['accuracy'],
            'Regular Accuracy': regular_history['accuracy']
        })

        # Save the data as CSV for reference
        df.to_csv(os.path.join(save_dir, 'training_curves_data.csv'), index=False)
        logger.info(f"Training data saved to {os.path.join(save_dir, 'training_curves_data.csv')}")

        # Create interactive Plotly figure
        fig = make_subplots(rows=1, cols=2, subplot_titles=('Training Loss', 'Training Accuracy'))

        # Add loss traces - explicitly convert to list
        fig.add_trace(
            go.Scatter(x=list(epochs), y=list(curriculum_history['loss']), mode='lines+markers',
                      name='Curriculum Loss', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=list(epochs), y=list(regular_history['loss']), mode='lines+markers',
                      name='Regular Loss', line=dict(color='red')),
            row=1, col=1
        )

        # Add accuracy traces - explicitly convert to list
        fig.add_trace(
            go.Scatter(x=list(epochs), y=list(curriculum_history['accuracy']), mode='lines+markers',
                      name='Curriculum Accuracy', line=dict(color='blue')),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=list(epochs), y=list(regular_history['accuracy']), mode='lines+markers',
                      name='Regular Accuracy', line=dict(color='red')),
            row=1, col=2
        )

        # Update layout
        fig.update_layout(
            height=600,
            width=1200,
            title_text="Training Curves: Curriculum vs Regular Learning",
            hovermode="x unified"
        )

        # Add range slider
        fig.update_xaxes(rangeslider_visible=True)

        # Save as interactive HTML
        html_path = os.path.join(save_dir, f'interactive_training_curves_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html')
        fig.write_html(html_path)
        logger.info(f"Interactive training curves saved to {html_path}")

        try:
            # Save as static image (doesn't require display)
            png_path = os.path.join(save_dir, f'training_curves_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
            fig.write_image(png_path)
            logger.info(f"Static training curves saved to {png_path}")
        except Exception as img_error:
            logger.error(f"Error saving static image: {str(img_error)}")

        # Also save as JSON for later loading
        json_path = os.path.join(save_dir, f'training_curves_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        fig.write_json(json_path)
        logger.info(f"Training curves data saved to {json_path}")

    except Exception as e:
        logger.error(f"Error plotting with Plotly: {str(e)}")

        # Fallback to matplotlib with non-interactive backend
        try:
            logger.info("Falling back to matplotlib for static plots")
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend

            plt.figure(figsize=(12, 5))

            plt.subplot(1, 2, 1)
            plt.plot(epochs, curriculum_history['loss'], 'b-', label='Curriculum Loss')
            plt.plot(epochs, regular_history['loss'], 'r-', label='Regular Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Loss')
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(epochs, curriculum_history['accuracy'], 'b-', label='Curriculum Accuracy')
            plt.plot(epochs, regular_history['accuracy'], 'r-', label='Regular Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Training Accuracy')
            plt.legend()

            plt.tight_layout()

            # Save figure
            fallback_path = os.path.join(save_dir, f'training_curves_fallback_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
            plt.savefig(fallback_path)
            plt.close()
            logger.info(f"Fallback training curves saved to {fallback_path}")

        except Exception as e2:
            logger.error(f"Error with fallback plotting: {str(e2)}")

# Main function to run the experiment
def run_experiment(args):
    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logger.info(f"Using device: {device}")

    # Set random seed for reproducibility
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
        logger.info(f"Random seed set to {args.seed}")

    try:
        # Curriculum Learning
        model_curriculum = SimpleCNN()
        optimizer_curriculum = torch.optim.Adam(model_curriculum.parameters(), lr=args.learning_rate)
        criterion = nn.CrossEntropyLoss()

        # Check for curriculum checkpoint
        curriculum_start_epoch = 0
        curriculum_history = {'loss': [], 'accuracy': []}
        curriculum_config = {}

        if args.resume_curriculum:
            model_curriculum, curriculum_start_epoch, curriculum_history, curriculum_config = load_checkpoint(
                args.resume_curriculum, model_curriculum, optimizer_curriculum
            )

            # If we have a config, use it to determine where to resume
            if curriculum_config and 'current_resolution_idx' in curriculum_config:
                current_resolution_idx = curriculum_config['current_resolution_idx']
                resolutions = args.resolutions[:current_resolution_idx+1]
                logger.info(f"Resuming curriculum training from resolution index {current_resolution_idx}")
            else:
                resolutions = args.resolutions
        else:
            resolutions = args.resolutions

        # Start or resume curriculum training
        if curriculum_start_epoch == 0 or not args.resume_curriculum:
            logger.info("Starting curriculum learning from scratch")
            for idx, resolution in enumerate(resolutions):
                try:
                    logger.info(f"\nCurriculum Learning: Training on resolution {resolution}x{resolution}")
                    dataloader = get_dataloader(resolution, batch_size=args.batch_size, num_workers=args.num_workers)

                    # Save current curriculum state in config
                    curriculum_config['current_resolution_idx'] = idx

                    stage_history = train_model(
                        model_curriculum,
                        dataloader,
                        optimizer_curriculum,
                        criterion,
                        args.epochs_per_stage,
                        device,
                        start_epoch=0,  # Each resolution starts from epoch 0
                        history={'loss': [], 'accuracy': []},  # Fresh history for this stage
                        checkpoint_dir=args.checkpoint_dir,
                        checkpoint_interval=args.checkpoint_interval,
                        model_name=f'curriculum_res{resolution}'
                    )

                    # Extend the overall curriculum history
                    curriculum_history['loss'].extend(stage_history['loss'])
                    curriculum_history['accuracy'].extend(stage_history['accuracy'])

                    # Save the combined history after each resolution
                    save_history_to_csv(
                        curriculum_history,
                        f'{args.output_dir}/curriculum_combined_history.csv'
                    )
                except Exception as e:
                    logger.error(f"Error during curriculum training at resolution {resolution}: {str(e)}")
                    # Save what we have so far
                    checkpoint = {
                        'epoch': 0,
                        'model_state_dict': model_curriculum.state_dict(),
                        'optimizer_state_dict': optimizer_curriculum.state_dict(),
                        'history': curriculum_history,
                        'config': curriculum_config
                    }
                    save_checkpoint(
                        checkpoint,
                        checkpoint_dir=args.checkpoint_dir,
                        filename=f'curriculum_error_res{resolution}.pth.tar'
                    )
                    if idx < len(resolutions) - 1:
                        logger.info(f"Attempting to continue with next resolution {resolutions[idx+1]}")
                        continue
                    else:
                        break
        else:
            logger.info(f"Resuming curriculum learning from epoch {curriculum_start_epoch}")
            # Continue from where we left off
            current_resolution_idx = curriculum_config.get('current_resolution_idx', 0)
            resolution = resolutions[current_resolution_idx]

            logger.info(f"Resuming curriculum training at resolution {resolution}x{resolution}")
            dataloader = get_dataloader(resolution, batch_size=args.batch_size, num_workers=args.num_workers)

            # Calculate remaining epochs
            remaining_epochs = args.epochs_per_stage - (curriculum_start_epoch % args.epochs_per_stage)

            if remaining_epochs > 0:
                stage_history = train_model(
                    model_curriculum,
                    dataloader,
                    optimizer_curriculum,
                    criterion,
                    remaining_epochs,
                    device,
                    start_epoch=curriculum_start_epoch,
                    history={'loss': [], 'accuracy': []},  # Fresh history for this stage
                    checkpoint_dir=args.checkpoint_dir,
                    checkpoint_interval=args.checkpoint_interval,
                    model_name=f'curriculum_res{resolution}'
                )

                # Extend the overall curriculum history
                curriculum_history['loss'].extend(stage_history['loss'])
                curriculum_history['accuracy'].extend(stage_history['accuracy'])

            # Continue with remaining resolutions
            for idx, resolution in enumerate(resolutions[current_resolution_idx+1:], start=current_resolution_idx+1):
                try:
                    logger.info(f"\nCurriculum Learning: Training on resolution {resolution}x{resolution}")
                    dataloader = get_dataloader(resolution, batch_size=args.batch_size, num_workers=args.num_workers)

                    # Update current resolution in config
                    curriculum_config['current_resolution_idx'] = idx

                    stage_history = train_model(
                        model_curriculum,
                        dataloader,
                        optimizer_curriculum,
                        criterion,
                        args.epochs_per_stage,
                        device,
                        start_epoch=0,  # Each resolution starts from epoch 0
                        history={'loss': [], 'accuracy': []},  # Fresh history for this stage
                        checkpoint_dir=args.checkpoint_dir,
                        checkpoint_interval=args.checkpoint_interval,
                        model_name=f'curriculum_res{resolution}'
                    )

                    # Extend the overall curriculum history
                    curriculum_history['loss'].extend(stage_history['loss'])
                    curriculum_history['accuracy'].extend(stage_history['accuracy'])

                    # Save the combined history after each resolution
                    save_history_to_csv(
                        curriculum_history,
                        f'{args.output_dir}/curriculum_combined_history.csv'
                    )
                except Exception as e:
                    logger.error(f"Error during curriculum training at resolution {resolution}: {str(e)}")
                    # Save what we have so far
                    checkpoint = {
                        'epoch': 0,
                        'model_state_dict': model_curriculum.state_dict(),
                        'optimizer_state_dict': optimizer_curriculum.state_dict(),
                        'history': curriculum_history,
                        'config': curriculum_config
                    }
                    save_checkpoint(
                        checkpoint,
                        checkpoint_dir=args.checkpoint_dir,
                        filename=f'curriculum_error_res{resolution}.pth.tar'
                    )
                    if idx < len(resolutions) - 1:
                        logger.info(f"Attempting to continue with next resolution {resolutions[idx+1]}")
                        continue
                    else:
                        break

        # Regular Supervised Learning
        model_regular = SimpleCNN()
        optimizer_regular = torch.optim.Adam(model_regular.parameters(), lr=args.learning_rate)

        # Check for regular training checkpoint
        regular_start_epoch = 0
        regular_history = {'loss': [], 'accuracy': []}

        if args.resume_regular:
            model_regular, regular_start_epoch, regular_history, _ = load_checkpoint(
                args.resume_regular, model_regular, optimizer_regular
            )

        # Calculate total epochs for regular training
        total_epochs = len(args.resolutions) * args.epochs_per_stage
        remaining_regular_epochs = total_epochs - regular_start_epoch

        if remaining_regular_epochs > 0:
            logger.info(f"\nRegular Supervised Learning: Training on resolution {args.final_resolution}x{args.final_resolution}")
            logger.info(f"Starting from epoch {regular_start_epoch+1}/{total_epochs}")

            dataloader_regular = get_dataloader(args.final_resolution, batch_size=args.batch_size, num_workers=args.num_workers)

            try:
                regular_history = train_model(
                    model_regular,
                    dataloader_regular,
                    optimizer_regular,
                    criterion,
                    remaining_regular_epochs,
                    device,
                    start_epoch=regular_start_epoch,
                    history=regular_history,
                    checkpoint_dir=args.checkpoint_dir,
                    checkpoint_interval=args.checkpoint_interval,
                    model_name='regular'
                )
            except Exception as e:
                logger.error(f"Error during regular training: {str(e)}")
                # Save what we have so far
                checkpoint = {
                    'epoch': regular_start_epoch,
                    'model_state_dict': model_regular.state_dict(),
                    'optimizer_state_dict': optimizer_regular.state_dict(),
                    'history': regular_history
                }
                save_checkpoint(
                    checkpoint,
                    checkpoint_dir=args.checkpoint_dir,
                    filename='regular_error.pth.tar'
                )

        

        # Compute Area Under the Curve (AUC) for accuracy correctly
        curriculum_auc = 0
        regular_auc = 0
        curriculum_auc_normalized = 0
        regular_auc_normalized = 0
        curriculum_test_acc, regular_test_acc = 0, 0

        # Initialize results dictionary before AUC calculation
        results = {
            'curriculum_test_accuracy': curriculum_test_acc,
            'regular_test_accuracy': regular_test_acc,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # Evaluate both models on test set with final resolution
        try:
            test_dataloader = get_dataloader(args.final_resolution, batch_size=args.batch_size, train=False, num_workers=args.num_workers)
            curriculum_test_acc = evaluate_model(model_curriculum, test_dataloader, device)
            regular_test_acc = evaluate_model(model_regular, test_dataloader, device)

            logger.info(f"\nCurriculum Learning Test Accuracy: {curriculum_test_acc:.4f}")
            logger.info(f"Regular Training Test Accuracy: {regular_test_acc:.4f}")

            # Save final models
            torch.save(model_curriculum.state_dict(), os.path.join(args.output_dir, 'curriculum_final_model.pth'))
            torch.save(model_regular.state_dict(), os.path.join(args.output_dir, 'regular_final_model.pth'))
            logger.info("Final models saved to output directory")

            try:
                # Ensure we're using the same x-axis range for both
                x_values = np.array(list(range(1, len(curriculum_history['accuracy']) + 1)))

                # Calculate AUC using trapezoidal rule
                curriculum_auc = trapz(np.array(curriculum_history['accuracy']), x=x_values)
                regular_auc = trapz(np.array(regular_history['accuracy']), x=x_values)

                # Normalize by the number of epochs for easier interpretation
                curriculum_auc_normalized = curriculum_auc / len(x_values)
                regular_auc_normalized = regular_auc / len(x_values)

                logger.info(f"Curriculum Learning AUC: {curriculum_auc:.4f} (normalized: {curriculum_auc_normalized:.4f})")
                logger.info(f"Regular Training AUC: {regular_auc:.4f} (normalized: {regular_auc_normalized:.4f})")

                # Also calculate AUC for the second half of training to see late-stage performance
                if len(x_values) >= 4:  # Only if we have enough epochs
                    half_point = len(x_values) // 2
                    curriculum_late_auc = trapz(np.array(curriculum_history['accuracy'][half_point:]),
                                               x=x_values[half_point:])
                    regular_late_auc = trapz(np.array(regular_history['accuracy'][half_point:]),
                                            x=x_values[half_point:])

                    # Normalize by the number of epochs
                    curriculum_late_auc_normalized = curriculum_late_auc / (len(x_values) - half_point)
                    regular_late_auc_normalized = regular_late_auc / (len(x_values) - half_point)

                    logger.info(f"Curriculum Learning Late-Stage AUC: {curriculum_late_auc:.4f} (normalized: {curriculum_late_auc_normalized:.4f})")
                    logger.info(f"Regular Training Late-Stage AUC: {regular_late_auc:.4f} (normalized: {regular_late_auc_normalized:.4f})")

                    # Add these to the results dictionary
                    results['curriculum_late_auc'] = float(curriculum_late_auc)
                    results['regular_late_auc'] = float(regular_late_auc)
                    results['curriculum_late_auc_normalized'] = float(curriculum_late_auc_normalized)
                    results['regular_late_auc_normalized'] = float(regular_late_auc_normalized)
            except Exception as e:
                logger.error(f"Error calculating AUC: {str(e)}")
                curriculum_auc = sum(curriculum_history['accuracy'])
                regular_auc = sum(regular_history['accuracy'])
                curriculum_auc_normalized = curriculum_auc / len(curriculum_history['accuracy'])
                regular_auc_normalized = regular_auc / len(regular_history['accuracy'])
                logger.info(f"Using fallback AUC calculation (simple sum)")
                logger.info(f"Curriculum Learning AUC (sum): {curriculum_auc:.4f} (normalized: {curriculum_auc_normalized:.4f})")
                logger.info(f"Regular Training AUC (sum): {regular_auc:.4f} (normalized: {regular_auc_normalized:.4f})")

            # Add AUC values to results dictionary
            results['curriculum_auc'] = float(curriculum_auc)
            results['regular_auc'] = float(regular_auc)
            results['curriculum_auc_normalized'] = float(curriculum_auc_normalized)
            results['regular_auc_normalized'] = float(regular_auc_normalized)

            # Find first epoch where accuracy exceeds threshold
            threshold = args.accuracy_threshold
            curriculum_first = first_epoch_above_threshold(curriculum_history, threshold)
            regular_first = first_epoch_above_threshold(regular_history, threshold)
            logger.info(f"Curriculum Learning first epoch above {threshold}: {curriculum_first}")
            logger.info(f"Regular Training first epoch above {threshold}: {regular_first}")

            # Add threshold information to results
            results['curriculum_first_epoch_above_threshold'] = curriculum_first
            results['regular_first_epoch_above_threshold'] = regular_first
            results['threshold'] = threshold
            results['curriculum_test_accuracy'] = curriculum_test_acc
            results['regular_test_accuracy'] = regular_test_acc

            # Save results to JSON
            with open(os.path.join(args.output_dir, 'experiment_results.json'), 'w') as f:
                json.dump(results, f, indent=4)
            logger.info(f"Results saved to {os.path.join(args.output_dir, 'experiment_results.json')}")

            plot_training_curves(curriculum_history, regular_history, total_epochs, save_dir=args.output_dir)


        except Exception as e:
            logger.error(f"Error during evaluation and results processing: {str(e)}")

    except KeyboardInterrupt:
        logger.info("Experiment interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error in experiment: {str(e)}")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Curriculum Learning vs Regular Training Experiment')

    # Training parameters
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs-per-stage', type=int, default=2, help='Number of epochs per resolution stage')
    parser.add_argument('--resolutions', type=int, nargs='+', default=[16, 24, 32], help='Resolutions for curriculum learning')
    parser.add_argument('--final-resolution', type=int, default=32, help='Final resolution for evaluation')
    parser.add_argument('--accuracy-threshold', type=float, default=0.5, help='Accuracy threshold for epoch counting')

    # System parameters
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disable CUDA training')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--num-workers', type=int, default=2, help='Number of workers for data loading')

    # Checkpoint parameters
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--output-dir', type=str, default='./output', help='Directory to save output files')
    parser.add_argument('--checkpoint-interval', type=int, default=1, help='Epochs between checkpoint saves')
    parser.add_argument('--resume-curriculum', type=str, default=None, help='Path to curriculum checkpoint to resume from')
    parser.add_argument('--resume-regular', type=str, default=None, help='Path to regular training checkpoint to resume from')

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging()

    # Log the arguments+
    logger.info(f"Running with arguments: {args}")

    # Run the experiment
    run_experiment(args)