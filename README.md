# Traffic Flow Prediction Using Echo State Networks (ESNs)

This repository is part of a dissertation project exploring Echo State Network (ESN) architectures for predicting traffic flow patterns. Starting with a basic ESN, this project aims to evolve into more complex models, such as stacked and SpaRCe ESNs, to capture the complex temporal dynamics inherent in traffic data.

## Project Structure

- **config/**: Configuration files for data paths, model parameters, and training settings.
    - `data_config.yaml`: Paths and preprocessing options for raw and processed data.
    - `model_config.yaml`: Model parameters, organized by ESN architecture (simple, stacked, advanced).
    - `train_config.yaml`: Training settings including batch size, learning rate, and early stopping.

- **data/**: Contains datasets for training and testing.
    - `raw/`: Raw traffic datasets.
    - `processed/`: Preprocessed datasets ready for model input.
    - `external/`: External datasets with details on their source and format.

- **models/**: Saved trained models, organized by architecture.
    - `simple/`: Basic ESN models.
    - `stacked/`: Stacked ESN models.
    - `advanced/`: Advanced ESN models for complex experiments.

- **results/**: Stores evaluation metrics, visualizations, and logs.
    - `model_evaluations/`: Performance metrics for each model and experiment.
    - `figures/`: Visualizations of results, such as training curves and prediction plots.
    - `logs/`: Detailed logs from training and evaluation runs.

- **src/**: Core code modules for data preprocessing, model definitions, and training scripts.
    - `data_preprocessing.py`: Functions for data cleaning, normalization, and transformation.
    - `simple_esn/`: Basic ESN model setup and training scripts.
    - `stacked_esn/`: Code for building and training a stacked ESN.
    - `advanced_esn/`: Advanced ESN architectures.

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

