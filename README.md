# FourierFlow Generator for Financial Time Series
The FourierFlow Generator is a simple, yet powerful tool for generating synthetic time series data using generative models.

# Getting Started
## Installation:
Clone this repository to your local machine.
I recommend to create a new environment by running
```console
conda env create -f environment.yml
```

## Configuration:
- Navigate to the config_ff.yaml file.
- Adjust the parameters according to your needs:
    + `input_dim`: Dimensionality of input features.
    + `output_dim`: Dimensionality of output features.
    + `hidden_dim`: Size of hidden layers.
    + `n_flows`: Number of flow layers.
    + `n_lags`: Number of time lags.
    + `vol_activation`: Activation function for volatility modeling (e.g., “softplus”).
    + Other hyperparameters (batch size, learning rate, etc.).
- Pretraining or Checkpoint:
    + Choose between two modes:
        + Pretrain:
            + Load your training data and labels (regular and crisis data).
            + Train the generator using generator_regular.fit() and generator_crisis.fit().
            + Save the combined model dictionary using save_combined_model_dict().
        + Checkpoint (Pickle Files):
            + Load the pre-trained model from model_dict.pkl.
- Generating Samples:
    + Run the script main.py in console:
    ```console
    python main.py
    ```
    + Specify the condition (e.g., crisis or regular) by setting condition[0] in the `main.py`.
    + The generated synthetic data will be saved to a pickle file.