import numpy as np
import os
import argparse
import pandas as pd
import time

def generate_narma(N=10, num_samples=2000, input_range=(0, 0.5), seed=None):
    """
    Generates a NARMA-N dataset and saves it in CSV format.

    Parameters:
    - N (int): Order of the NARMA system (e.g., 10 for NARMA-10).
    - num_samples (int): Total number of time steps in the dataset.
    - input_range (tuple): Range for generating random input values (default: (0, 0.5)).
    - seed (int, optional): Random seed for reproducibility. If None, a random seed is generated.

    Returns:
    - u (numpy array): Generated input sequence.
    - x (numpy array): Corresponding output sequence.
    - seed (int): The seed used for random number generation.
    """
    if seed is None:
        seed = np.random.randint(0, 2**31 - 1) # Generate a random seed if none is provided
        generated_random_seed = True
    else:
        generated_random_seed = False

    np.random.seed(seed)

    # Generate random input sequence u(t) in the given range
    u = np.random.uniform(input_range[0], input_range[1], num_samples)

    # Initialize output sequence x(t) with zeros
    x = np.zeros(num_samples)

    # Define NARMA equation coefficients
    alpha = 0.3
    beta = 0.05
    gamma = 1.5
    delta = 0.1

    # Generate the NARMA-N sequence
    for t in range(N, num_samples - 1):
        x[t + 1] = (alpha * x[t] +
                    beta * x[t] * np.sum(x[t - N + 1:t + 1]) +
                    gamma * u[t - N + 1] * u[t] +
                    delta)

    return u, x, seed, generated_random_seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate NARMA-N Dataset")
    parser.add_argument("--N", type=int, default=10, help="Order of NARMA (default: 10)")
    parser.add_argument("--num_samples", type=int, default=2000, help="Number of time steps (default: 2000)")
    parser.add_argument("--output_dir", type=str, default="data/benchmark/", help="Directory to save the dataset")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default: None, generates a random seed)")
    args = parser.parse_args()

    # Generate dataset
    u, x, seed, random_seed_generated = generate_narma(N=args.N, num_samples=args.num_samples, seed=args.seed)

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Create dataset filename
    filename = f"NARMA{args.N}_{args.num_samples}_seed{seed}.csv"
    filepath = os.path.join(args.output_dir, filename)

    # Save dataset as CSV
    df = pd.DataFrame({"Input": u, "Output": x})
    df.to_csv(filepath, index=False)

    # Log the seed if it was randomly generated
    log_file = os.path.join(args.output_dir, "seed_log.txt")
    if random_seed_generated:
        with open(log_file, "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - NARMA{args.N} with {args.num_samples} samples used seed: {seed}\n")

    print(f"Dataset saved: {filepath}")
    if random_seed_generated:
        print(f"Random seed {seed} was generated and logged in {log_file}")
    else:
        print(f"Used provided seed: {seed}")
