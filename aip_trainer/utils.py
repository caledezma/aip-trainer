"""Utility functions"""
import argparse
import os

from typing import Optional

import pandas as pd
import tensorflow as tf
import wandb

from google.cloud import secretmanager
from sklearn.model_selection import train_test_split

def get_args() -> argparse.ArgumentParser:
    """Get the arguments required for training the ML model"""
    parser = argparse.ArgumentParser(
        description="Train a ML model"
    )
    parser.add_argument(
        "--dropout",
        type=int,
        default=0.3,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--hidden-layers",
        type=int,
        default=3
    )
    parser.add_argument(
        "--hidden-units",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--reg-coeff",
        type=int,
        default=0.1,
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="categorical_crossentropy"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--use-wandb-secret",
        action="store_true",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="carlos-aip-tests",
    )
    return parser


def get_ml_model(
    input_size: int,
    hidden_layers: int,
    hidden_units: int,
    reg_coeff: float,
    dropout: float,
    output_size: int,
    loss: str,
    optimizer: str,
) -> tf.keras.Model:
    """Return a compiled Keras MLP the hyperparameters specified"""
    inputs = tf.keras.Input(shape=(input_size,))
    output = inputs
    for _ in range(hidden_layers):
        output = tf.keras.layers.Dense(
            units=hidden_units,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(reg_coeff),
        )(output)
        output = tf.keras.layers.BatchNormalization()(output)
        output=tf.keras.layers.Dropout(dropout)(output)
    output = tf.keras.layers.Dense(
        units=output_size,
        activation="softmax",
        kernel_regularizer=tf.keras.regularizers.l2(reg_coeff),
    )(output)
    model = tf.keras.Model(inputs, output)
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy'],
        
    )
    return model


def get_wine_data(filepath, test_size):
    """Get and prepare the wine data for ML"""
    dataset = pd.read_csv(filepath)
    num_classes = dataset.quality.max() + 1
    dataset=dataset.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        dataset[:,:-2], dataset[:,-2],
        test_size=test_size,
    )

    return (
        X_train,
        X_test,
        tf.keras.utils.to_categorical(y_train, num_classes=num_classes),
        tf.keras.utils.to_categorical(y_test, num_classes=num_classes)
    )


def initialise_wandb(project: str, run_name: Optional[str] = None, use_secret=False) -> bool:
    """Initialise weights and biases and return the run name"""
    try:
        if use_secret:
            secrets_client = secretmanager.SecretManagerServiceClient()
            wandb_secret_location = os.environ.get("WANDB_SECRET")
            if wandb_secret_location is None:
                raise ValueError("Couldn't retrieve location of WandB secret")
            client_response = secrets_client.access_secret_version(
                request={"name": wandb_secret_location}
            )
            api_key = client_response.payload.data.decode("UTF-8")
            os.environ["WANDB_API_KEY"] = api_key
        
        if os.environ.get("WANDB_API_KEY") is None:
            raise ValueError("Either use the secret or specify a key in WANDB_API_KEY env var")
        wandb.init(
            name=run_name,
            project=project,
        )
        print(f"Logging to W&B under project {wandb.run.project} and run {wandb.run.name}")
        return True
    except Exception as exc:
        print("Could not initialise W&B due to")
        print(exc)
        return False
