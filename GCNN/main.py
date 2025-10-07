import torch
from torch.nn import ReLU
from model.operators import normalized_hub_laplacian
from experiments.gcnn_train import run_experiment
import itertools
import os
import pandas as pd
from utils.gs_utils import generate_run_id, plot_val_mae_per_target, plot_alphas_history
import csv

def get_config_grid():
    """Defines and returns the grid of configurations to be tested."""
    default_params = {
        "N": 1200, "targets": [0], "batch_size": 64, "lr": 1e-4,
        "alpha_lr": 1e-2, "weight_decay": 1e-5, "alpha": 0.5, "num_epochs": 300,
        "dims": [11, 64, 64], "hops": 2, "act_fn": ReLU(),
        "readout_hidden_dims": [64, 32], "pooling": "mean", "apply_readout": True,
        "learn_alpha": True, "gso_generator": normalized_hub_laplacian,
        "use_bn": True, "dropout_p": 0.2, "patience": 50
    }
    grid_params = {"learn_alpha": [False, True], "alpha": [-0.5, 0, 0.5, 1]}

    keys, values = zip(*grid_params.items())
    grid = [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    configs = []
    for g_params in grid:
        config = default_params.copy()
        config.update(g_params)
        configs.append(config)
    return configs

def run_and_log_trial(config, results_dir):
    """Runs a single experiment trial and logs the results."""
    run_id = generate_run_id(4)
    run_dir = os.path.join(results_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n--- Starting Run ID: {run_id} ---")

    model, best_val, test_mae, val_mae_hist, alphas_hist = run_experiment(config)

    torch.save(model.state_dict(), os.path.join(run_dir, "model.pt"))
    plot_val_mae_per_target(val_mae_hist, run_id, save_dir=run_dir)
    plot_alphas_history(alphas_hist, run_id, save_dir=run_dir)

    row = {"run_id": run_id, "best_val_mean_mae": best_val, "mean_test_mae": test_mae.mean().item()}
    for i, mae in enumerate(test_mae):
        row[f"test_target_{i}_mae"] = mae.item()

    for name, value in config.items():
        row[name] = value.__name__ if callable(value) else str(value)

    print(f"✅ Run {run_id} complete. Artifacts saved to {run_dir}")
    return row

def summarize_results(results, output_file):
    """Writes the summary of all trial results to a CSV file."""
    if not results:
        return

    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)

    print(f"\nAll runs complete. Summary written to {output_file}")
    best_run = df.loc[df["mean_test_mae"].idxmin()]
    print("\n--- Best Run (lowest mean_test_mae) ---")
    print(best_run.to_string())

def main():
    """Main function to run the grid search and save results."""
    results_dir = "GCNN/results_turbo"
    os.makedirs(results_dir, exist_ok=True)

    configs = get_config_grid()
    all_results = []

    for i, config in enumerate(configs, 1):
        print(f"\n--- Running Trial {i}/{len(configs)} ---")
        trial_results = run_and_log_trial(config, results_dir)
        all_results.append(trial_results)

    summary_file = os.path.join(results_dir, "summary.csv")
    summarize_results(all_results, summary_file)

if __name__ == "__main__":
    main()
