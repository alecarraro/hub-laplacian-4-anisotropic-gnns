import os
import random
import string
import matplotlib.pyplot as plt
import torch

def generate_run_id(length=6):
    """Generates a random run ID."""
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))

def plot_val_mae_per_target(val_per_target_mae_history, run_id, save_dir="grid_search_results"):
    """
    Plots and saves per-target validation MAE curves.
    Args:
        val_per_target_mae_history (list of lists): Each entry is a list of per-target MAEs for that epoch.
        run_id (str): Unique identifier for the run.
        save_dir (str): Directory to save the plot.
    """
    os.makedirs(save_dir, exist_ok=True)
    epochs = list(range(1, len(val_per_target_mae_history) + 1))
    val_per_target_mae_array = list(map(list, zip(*val_per_target_mae_history)))

    plt.figure(figsize=(10, 6))
    for idx, target_mae in enumerate(val_per_target_mae_array):
        plt.plot(epochs, target_mae, label=f"Val MAE - Target {idx}")

    plt.xlabel("Epoch")
    plt.ylabel("Validation MAE")
    plt.title(f"Validation MAE per Target (Run {run_id})")
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"loss_{run_id}.png"))
    plt.close()

def plot_alphas_history(alphas_history, run_id, save_dir="grid_search_results"):
    """
    Plots and saves the history of alpha values over epochs.
    Args:
        alphas_history (list): A list of alpha values, one for each epoch.
        run_id (str): Unique identifier for the run.
        save_dir (str): Directory to save the plot.
    """
    os.makedirs(save_dir, exist_ok=True)
    epochs = list(range(1, len(alphas_history) + 1))

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, alphas_history, label="Alpha Value")
    plt.xlabel("Epoch")
    plt.ylabel("Alpha")
    plt.title(f"Alpha Value Over Epochs (Run {run_id})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"alpha_{run_id}.png"))
    plt.close()