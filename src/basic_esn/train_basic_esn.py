import numpy as np
import os
import json
import datetime
from src.basic_esn.basic_esn_model import BasicESN
from src.utils.test_esn import test_esn
from src.utils.logger import create_experiment_dir, save_experiment_results, save_predictions_plot

# Define dataset name manually
dataset_name = "NARMA10_3000_seed976071145" 

# Define parameter grid
spectral_radius_values = [0.5, 0.8, 1.0, 1.2, 1.5]
leaking_rate_values = [0.1, 0.3, 0.5, 0.8, 1.0]
input_scaling_values = [0.01, 0.05, 0.1, 0.5]
reservoir_sizes = [50, 100, 200]

# Create a new experiment log file
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
experiment_log_dir = f"results/model_evaluations/benchmarks/{dataset_name}/"
os.makedirs(experiment_log_dir, exist_ok=True)
experiment_log_file = os.path.join(experiment_log_dir, f"experiment_{timestamp}.txt")

# Open log file for writing
with open(experiment_log_file, "w", encoding="utf-8") as log:
    log.write(f"===== Experiment Run ({timestamp}) on {dataset_name} =====\n")
    
    # Perform Grid Search
    best_nrmse = float("inf")
    best_params = {}

    for spectral_radius in spectral_radius_values:
        for leaking_rate in leaking_rate_values:
            for input_scaling in input_scaling_values:
                for reservoir_size in reservoir_sizes:
                    # Initialize ESN
                    esn = BasicESN(
                        input_size=1,
                        reservoir_size=reservoir_size,
                        output_size=1,
                        target_spectral_radius=spectral_radius,
                        leaking_rate=leaking_rate,
                        input_scaling=input_scaling
                    )

                    # Run ESN on dataset
                    dataset_path = f"data/benchmark/{dataset_name}.csv"
                    nrmse, y_true, y_pred = test_esn(esn, dataset_path=dataset_path)

                    log_message = f"ρ(W)={spectral_radius}, α={leaking_rate}, Win scale={input_scaling}, Res Size={reservoir_size} → NRMSE: {nrmse}\n"
                    print(log_message)
                    log.write(log_message)

                    # Save best configuration
                    if nrmse < best_nrmse:
                        best_nrmse = nrmse
                        best_params = {
                            "spectral_radius": spectral_radius,
                            "leaking_rate": leaking_rate,
                            "input_scaling": input_scaling,
                            "reservoir_size": reservoir_size
                        }

    # Log the best results at the end of the experiment
    log.write("\n===== Best Parameters for This Experiment =====\n")
    log.write(json.dumps(best_params, indent=4) + "\n")
    log.write(f"Best NRMSE: {best_nrmse}\n")
    log.write("=" * 50 + "\n")

print(f"\nExperiment log saved: {experiment_log_file}")
print("\nBest Parameters for This Experiment:")
print(best_params)
print(f"Best NRMSE: {best_nrmse}")
