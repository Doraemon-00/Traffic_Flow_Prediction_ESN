import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from src.basic_esn.basic_esn_model import BasicESN  # Import ESN class

def test_esn(esn, dataset_path, alpha_ridge=1e-6):
    """
    Tests an ESN on the NARMA10 dataset and returns the NRMSE.

    Parameters:
    - esn: The ESN instance.
    - dataset_path: Path to the NARMA dataset.
    - alpha_ridge: Regularization strength for Ridge regression.

    Returns:
    - NRMSE (Normalized Root Mean Squared Error)
    - y_true: Ground truth output sequence
    - y_pred: ESN predictions
    """
    # Load dataset
    df = pd.read_csv(dataset_path)
    u = df["Input"].values.reshape(-1, 1)  # Ensure input is 2D
    x_target = df["Output"].values

    # Run ESN on input sequence
    reservoir_states = []
    for i in range(len(u) - 1):
        esn.update(u[i])
        reservoir_states.append(esn.x.copy())  # Store reservoir state

    # Convert to NumPy array
    X = np.array(reservoir_states)
    Y = x_target[1:]  # Target is next-step output

    # Train readout layer using Ridge regression
    ridge = Ridge(alpha=alpha_ridge)
    ridge.fit(X, Y)
    esn.Wout = ridge.coef_

    # Predict outputs
    Y_pred = X @ esn.Wout.T

    # Compute NRMSE
    nrmse = np.sqrt(np.mean((Y - Y_pred) ** 2)) / np.std(Y)
    
    return nrmse, Y, Y_pred  # Return predictions for logging
