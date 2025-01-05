import torch
import numpy as np
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.utils as utils
import time
import wandb  # Optional for logging
from evaluate import *
from utils import epoch_time
import os

os.environ["WANDB_ARTIFACT_LOCATION"] = "your_WANDB_ARTIFACT_LOCATION_pth"
os.environ["WANDB_ARTIFACT_DIR"] = "your_WANDB_ARTIFACT_DIR_pth"
os.environ["WANDB_CACHE_DIR"] = "your_WANDB_CACHE_DIR_pth"


class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def log_gradient_flow(named_parameters):
    """
    Logs the gradient flow of the model parameters.
    """
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if p.requires_grad and ("bias" not in n) and p.grad is not None:
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().item())
    return layers, ave_grads

def log_parameter_distributions(model):
    """
    Logs histograms of model parameters and their gradients.
    """
    distributions = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            distributions[f"param_{name}"] = wandb.Histogram(param.data.cpu())
            if param.grad is not None:
                distributions[f"grad_{name}"] = wandb.Histogram(param.grad.cpu())
    return distributions


def train_step(model, batch, clusters, criterion, gamma=1.0):
    """
    Performs a single training step on one batch of data.

    Args:
        model: The Seq2Seq model
        batch: Tuple of (src, src_len, trg)
        clusters: Clustering module
        criterion: Loss function for prediction
        gamma: Weight for clustering loss (default: 1.0)

    Returns:
        tuple: Total loss, cluster loss, and prediction loss
    """
    src, src_len, trg = batch

    # Forward pass
    output, enc_hidden = model(src, src_len, trg)
    trg = trg[:, 1]  # Using your original target processing

    # Calculate clustering loss
    q = clusters(enc_hidden)
    p = clusters.target_distribution(q)
    cluster_loss = F.kl_div(q.log(), p, reduction="batchmean") * (10 ** 9)

    # Calculate prediction loss
    pred_loss = criterion(output, trg)

    # Combined loss
    total_loss = gamma * cluster_loss + pred_loss

    # Log predictions for a small subset of the batch
    sample_preds = {
        'outputs': output[:3].detach().cpu(),  # First 3 predictions
        'targets': trg[:3].detach().cpu(),
        'hidden_states': enc_hidden[:3].detach().cpu()
    }

    return total_loss, cluster_loss, pred_loss, sample_preds


def train_epoch(model, train_loader, optimizer, clusters, criterion, max_grad_norm=1.0, gamma=1.0):
    model.train()
    total_train_loss = 0
    total_cluster_loss = 0
    total_pred_loss = 0

    gradient_norms = []
    grad_flows = []  # Store multiple gradient flows

    accum_steps = 1
    optimizer.zero_grad()

    for i, batch in enumerate(train_loader):
        train_loss, cluster_loss, pred_loss, sample_preds = train_step(
            model, batch, clusters, criterion, gamma
        )

        loss = train_loss / accum_steps
        loss.backward()

        # Log gradients before clipping
        if i % 100 == 0:  # Periodic logging to save memory
            layers, batch_grads = log_gradient_flow(model.named_parameters())
            grad_flows.append(batch_grads)

        # Single gradient clipping
        total_norm = utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        gradient_norms.append(total_norm.item())

        if (i + 1) % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_train_loss += train_loss.item()
        total_cluster_loss += cluster_loss.item()
        total_pred_loss += pred_loss.item()

        if i % 100 == 0:
            wandb.log({
                "batch/train_loss": train_loss.item(),
                "batch/cluster_loss": cluster_loss.item(),
                "batch/pred_loss": pred_loss.item(),
                "batch/gradient_norm": total_norm.item()
            })

    grad_stats = {
        "gradient_norm_mean": np.mean(gradient_norms),
        "gradient_norm_std": np.std(gradient_norms),
        "gradient_norm_max": np.max(gradient_norms)
    }

    # Average gradient flows across logged steps
    avg_gradients = np.mean(grad_flows, axis=0) if grad_flows else None

    return (
        total_train_loss / len(train_loader),
        total_cluster_loss / len(train_loader),
        total_pred_loss / len(train_loader),
        grad_stats,
        sample_preds,  # Just return last batch
        layers,
        avg_gradients  # Return averaged gradients
    )


def train(model, train_loader, val_loader, test_loader, criterion,
          clusters, epochs=100, patience=10, model_save_pth='best_model.pt',
          gamma=1.0, max_grad_norm=1.0):
    """
    Complete training procedure.
    """
    # Initialize wandb
    wandb.init(
        mode="offline",
        project='project_holic',
        config={
            "epochs": epochs,
            "patience": patience,
            "gamma": gamma,
            "max_grad_norm": max_grad_norm,
            "optimizer": "Adam",
            "weight_decay": 1e-5,
            "initial_lr": 0.001,
            "model_architecture": model.__class__.__name__,
            "criterion": criterion.__class__.__name__,
            "scheduler": "ReduceLROnPlateau"
        }
    )

    # Watch model for parameter and gradient tracking
    wandb.watch(model, log="all", log_freq=100)
    # Initialize optimizer with weight decay
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2,
                                  verbose=True, min_lr=1e-6)

    # Early stopping
    early_stopping = EarlyStopping(patience=patience)

    # Track metrics
    best_valid_loss = float('inf')

    for epoch in range(epochs):
        start_time = time.time()

        # Training phase
        train_loss, cluster_loss, pred_loss, grad_stats, samples, layers, gradients = train_epoch(
            model, train_loader, optimizer, clusters, criterion,
            max_grad_norm=max_grad_norm, gamma=gamma
        )

        # Validation phase
        model.eval()
        with torch.no_grad():
            valid_loss, epoch_mrr, epoch_recall = evaluate(
                model, val_loader, criterion
            )

        # Update learning rate
        scheduler.step(valid_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Log parameter distributions and gradients
        param_distributions = log_parameter_distributions(model)
        # layers, gradients = log_gradient_flow(model.named_parameters())

        # Comprehensive logging
        wandb.log({
            # Main metrics
            "epoch": epoch,
            "train/loss": train_loss,
            "train/cluster_loss": cluster_loss,
            "train/pred_loss": pred_loss,
            "val/loss": valid_loss,
            "val/mrr": epoch_mrr,
            "val/recall": epoch_recall,
            "learning_rate": current_lr,

            # Gradient statistics
            "gradients/mean_norm": grad_stats["gradient_norm_mean"],
            "gradients/std_norm": grad_stats["gradient_norm_std"],
            "gradients/max_norm": grad_stats["gradient_norm_max"],

            # Parameter distributions
            **param_distributions,

            # Gradient flow visualization
            "gradients/flow": wandb.plot.line_series(
                xs=range(len(layers)),
                ys=[gradients],
                keys=["gradient_magnitude"],
                title="Gradient Flow",
                xname="Layers"
            )
        })

        # Early stopping check
        early_stopping(valid_loss)

        # Early stopping and model saving
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'valid_loss': valid_loss,
            }, model_save_pth)

            # Log best model as artifact
            artifact = wandb.Artifact(
                name=f"model-epoch-{epoch}",
                type="model",
                description=f"Model checkpoint from epoch {epoch}"
            )
            artifact.add_file(model_save_pth)
            wandb.log_artifact(artifact)


        # Log progress
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(
                f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s'
                f' | Train Loss: {train_loss:.3f} | Cluster Loss: {cluster_loss:.3f}'
                f' | Val. Loss: {valid_loss:.3f} | Pred Loss: {pred_loss:.3f}'
                f' | MRR: {epoch_mrr:.3f} | Recall: {epoch_recall:.3f}'
                f' | Learning rate: {current_lr:.6f}'
              )

        if early_stopping.early_stop:
            print("Early stopping triggered")
            wandb.run.summary["stopped_epoch"] = epoch
            break

    # Load best model and evaluate on test set
    checkpoint = torch.load(model_save_pth)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_mrr, test_recall = evaluate(model, test_loader, criterion)
    print(f'Test Loss: {test_loss:.3f} | Test MRR: {test_mrr:.3f} | Test Recall: {test_recall:.3f}')

    # Log final metrics
    wandb.run.summary.update({
        "best_valid_loss": best_valid_loss,
        "test_loss": test_loss,
        "test_mrr": test_mrr,
        "test_recall": test_recall,
        "total_epochs": epoch + 1
    })

    wandb.finish()

    return model


""""
    - experimental
    - explore different approaches to leverage clusterig loss
"""

# dynamically balances the clustering loss and prediction loss by using their running averages
def train_lwn(model, iterator, optimizer, clusters, criterion, clip, gamma=1.0):
    """
    Trains the model for one epoch with loss weight normalization.

    Args:
        model: The Holic model to train.
        iterator: DataLoader providing the training batches.
        optimizer: Optimizer for the model.
        clusters: Clustering module for calculating cluster loss.
        criterion: Loss function for prediction loss.
        clip: Gradient clipping value.
        gamma: Weight for the clustering loss (default: 1.0).

    Returns:
        tuple: Average total loss, average cluster loss, and average prediction loss for the epoch.
    """
    model.train()

    # Initialize running averages for loss normalization
    cluster_running_avg = 1.0
    pred_running_avg = 1.0
    smoothing_factor = 0.99  # Weight for exponential moving average

    epoch_loss = 0.0
    epoch_cluster_loss = 0.0
    epoch_pred_loss = 0.0

    for src, src_len, trg in iterator:
        optimizer.zero_grad()

        # Forward pass
        output, p, q = model(src, src_len, trg)
        trg = trg[:, 1]

        # Calculate clustering loss

        cluster_loss = F.kl_div(q.log(), p, reduction="batchmean")

        # Calculate prediction loss
        pred_loss = criterion(output, trg)

        # Update running averages
        cluster_running_avg = (
            smoothing_factor * cluster_running_avg + (1 - smoothing_factor) * cluster_loss.item()
        )
        pred_running_avg = (
            smoothing_factor * pred_running_avg + (1 - smoothing_factor) * pred_loss.item()
        )

        # Normalize losses
        cluster_scaling_factor = pred_running_avg / (cluster_running_avg + 1e-8)
        normalized_cluster_loss = cluster_loss * cluster_scaling_factor

        # Combined loss
        loss = gamma * normalized_cluster_loss + pred_loss

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        # Optimizer step
        optimizer.step()

        # Accumulate losses
        epoch_loss += loss.item()
        epoch_cluster_loss += cluster_loss.item()
        epoch_pred_loss += pred_loss.item()

    # Return average losses
    num_batches = len(iterator)
    return (
        epoch_loss / num_batches,
        epoch_cluster_loss / num_batches,
        epoch_pred_loss / num_batches,
    )

# train with dynamic scaling
# using gradients for balancing the clustering loss and prediction loss
def train_ds(model, iterator, optimizer, clusters, criterion, clip):
    """
    Trains the model for one epoch using dynamic scaling of losses without a static scaling constant.

    Args:
        model: The Holic model to train.
        iterator: DataLoader providing the training batches.
        optimizer: Optimizer for the model.
        clusters: Clustering module for calculating cluster loss.
        criterion: Loss function for prediction loss.
        clip: Gradient clipping value.

    Returns:
        tuple: Average total loss, average cluster loss, and average prediction loss for the epoch.
    """
    model.train()

    epoch_loss = 0.0
    epoch_cluster_loss = 0.0
    epoch_pred_loss = 0.0

    for src, src_len, trg in iterator:
        optimizer.zero_grad()

        # Forward pass
        output, p, q = model(src, src_len, trg)
        trg = trg[:, 1]

        # Calculate clustering loss
        # q = clusters(enc_hidden)
        # p = clusters.target_distribution(q)
        cluster_loss = F.kl_div(q.log(), p, reduction="batchmean")

        # Calculate prediction loss
        pred_loss = criterion(output, trg)

        # Compute gradients for cluster_loss
        cluster_loss.backward(retain_graph=True)
        cluster_grad_norm = sum(p.grad.norm() for p in model.parameters() if p.grad is not None)

        # Clear gradients
        for p in model.parameters():
            if p.grad is not None:
                p.grad.zero_()

        # Compute gradients for pred_loss
        pred_loss.backward(retain_graph=True)
        pred_grad_norm = sum(p.grad.norm() for p in model.parameters() if p.grad is not None)

        # Clear gradients again
        for p in model.parameters():
            if p.grad is not None:
                p.grad.zero_()

        # Dynamically scale clustering loss
        scaling_factor = pred_grad_norm / (cluster_grad_norm + 1e-8)
        scaled_cluster_loss = cluster_loss * scaling_factor

        # Combine losses
        combined_loss = scaled_cluster_loss + pred_loss

        # Backpropagation on the combined loss
        combined_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        # Optimizer step
        optimizer.step()

        # Accumulate losses
        epoch_loss += combined_loss.item()
        epoch_cluster_loss += cluster_loss.item()
        epoch_pred_loss += pred_loss.item()

    # Return average losses
    num_batches = len(iterator)
    return (
        epoch_loss / num_batches,
        epoch_cluster_loss / num_batches,
        epoch_pred_loss / num_batches,
    )
