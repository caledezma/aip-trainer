"""Training entrypoint"""
from dotenv import find_dotenv, load_dotenv

import google.auth
from aip_trainer.utils import get_ml_model, get_args, get_wine_data, initialise_wandb
import wandb

load_dotenv(find_dotenv(usecwd=True))

def main():
    """Training code"""
    creds, project = google.auth.default()
    print(f"Project = {project}, Token = {creds.token}, expiry = {creds.expiry}, valid={creds.valid}")
    parser = get_args()
    args = parser.parse_args()
    wandb_active = initialise_wandb(
        project=args.wandb_project,
        run_name=args.wandb_run_name,
        use_secret=args.use_wandb_secret
    )
    
    X_train, X_test, y_train, y_test = get_wine_data("data/wine_data.csv", test_size=args.test_size)
    classifier = get_ml_model(
        input_size=X_train.shape[1],
        hidden_layers=args.hidden_layers,
        hidden_units=args.hidden_units,
        reg_coeff=args.reg_coeff,
        dropout=args.dropout,
        output_size=y_train.shape[1],
        loss=args.loss,
        optimizer=args.optimizer,
    )
    callbacks = []
    if wandb_active:
        callbacks.append(wandb.keras.WandbCallback(save_model=False))
    classifier.fit(
        x=X_train,
        y=y_train,
        epochs=args.epochs,
        validation_split=0.1,
        shuffle=True,
        callbacks=callbacks,
        verbose=2,
    )
    classifier.evaluate(
        x=X_test,
        y=y_test
    )


if __name__ == "__main__":
    """Script entrypoint"""
    main()
