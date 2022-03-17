"""Training entrypoint"""
from dotenv import find_dotenv, load_dotenv
import hypertune

from aip_trainer.utils import get_ml_model, get_args, get_wine_data

load_dotenv(find_dotenv(usecwd=True))

def main():
    """Training code"""
    parser = get_args()
    args = parser.parse_args()
    
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
    classifier.fit(
        x=X_train,
        y=y_train,
        epochs=args.epochs,
        validation_split=0.1,
        shuffle=True,
        verbose=2,
    )
    if args.save_location is not None:
        classifier.save(args.save_location)

    if args.hypertune:
        test_set_loss = classifier.evaluate(
            x=X_test,
            y=y_test,
            verbose=2,
        )[0]
        hpt = hypertune.HyperTune()
        hpt.report_hyperparameter_tuning_metric(
            hyperparameter_metric_tag="test_loss",
            metric_value=test_set_loss,
            global_step=int(args.epochs)
        )

if __name__ == "__main__":
    """Script entrypoint"""
    main()
