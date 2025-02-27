import os
import json
import datetime
import matplotlib.pyplot as plt

def create_experiment_dir(base_dir, dataset_name):
    """
    Creates a structured directory for experiment logging.

    - base_dir: The main results directory (e.g., "results/model_evaluations/benchmarks")
    - dataset_name: The dataset being used (e.g., "NARMA10_2000_seed923813934")

    Returns: Path to the experiment directory.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_dir, dataset_name, f"experiment_{timestamp}")
    
    os.makedirs(exp_dir, exist_ok=True)
    
    return exp_dir

def save_experiment_results(exp_dir, params, results):
    """
    Saves hyperparameters and results as JSON.
    """
    with open(os.path.join(exp_dir, "params.json"), "w") as f:
        json.dump(params, f, indent=4)
    
    with open(os.path.join(exp_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4)

def save_predictions_plot(exp_dir, y_true, y_pred):
    """
    Saves a comparison plot of predictions vs actual values.
    """
    plt.figure()
    plt.plot(y_true, label="True Values")
    plt.plot(y_pred, label="Predicted Values", linestyle="dashed")
    plt.legend()
    plt.title("Predictions vs Actual")
    plt.savefig(os.path.join(exp_dir, "predictions_vs_actual.png"))
    plt.close()
