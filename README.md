# Traffic Flow Prediction Using Echo State Networks (ESNs)

This repository is part of a dissertation project exploring Echo State Networks (ESNs) for traffic flow prediction. The project begins with a basic ESN, followed by Hierarchical ESN (HESN) and HESN with Sparse Learning (HESN-SL) to improve model performance and computational efficiency.

## Project Structure

- **config/**: Configuration files for data paths, model parameters, and training settings.
    - `data_config.yaml`: Paths and preprocessing options for raw and processed data.
    - `model_config.yaml`: Model parameters, organized by ESN architecture (simple, stacked, advanced).
    - `train_config.yaml`: Training settings including batch size, learning rate, and early stopping.

- **data/**: Contains datasets for training and testing.
    - `raw/`: Raw traffic datasets.
    - `processed/`: Preprocessed datasets ready for model input.
    - `benchmark/`: Synthetic datasets for benchmarks (NARMA10, Mackey-Glass).
    - `external/`: External datasets with details on their source and format.

- **models/**: Saved trained models, organized by architecture.
    - `basic_esn/`: Basic ESN models.
    - `hesn/`: Hierarchical ESN models.
    - `hesn_sl/`: HESN with Sparse Learning.

- **results/**: Stores evaluation metrics, visualizations, and logs.
    - `model_evaluations/`: Performance metrics for each model and experiment.
    - `figures/`: Visualizations of results, such as training curves and prediction plots.
    - `logs/`: Detailed logs from training and evaluation runs.

- **src/**: Core code modules for data preprocessing, model definitions, and training scripts.
    - `benchmarks/`: Scripts for running benchmark tests.
    - `data_preprocessing.py`: Functions for data cleaning, normalization, and transformation.
    - `basic_esn/`: Basic ESN model setup.
    - `hesn/`: Hierarchical ESN implementation.
    - `hesn_sl/`: HESN-SL training and evaluation.

## Getting Started

1. **Clone the Repository**:
    ```
    git clone https://github.com/Doraemon-00/Traffic_Flow_Prediction_ESN.git
    cd Traffic_Flow_Prediction_ESN
    ```

2. **Install Requirements**:
    ```
    pip install -r requirements.txt
    ```

3. **Set Up Data and Configuraitons**:  
    Place datasets in `data/raw/` or download from external sources to `data/external/`.  
    Adjust configurations as needed in `config/`.

4. **Run Experiments**:  
    Use scripts in `src/` to preprocess data, train models, and evaluate results.

