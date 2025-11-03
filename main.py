import os
import sys
import argparse
from src.train_model import train_bagged_models
from src.prediction import run_prediction


def main():
    parser = argparse.ArgumentParser(
        description="Train or run prediction for bagged site classifier models."
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train command
    train_parser = subparsers.add_parser("train", help="Train bagged models")
    train_parser.add_argument("dataset", type=str, help="Dataset file name (e.g. dataset0.json.gz)")
    train_parser.add_argument("label_file", type=str, help="Label file name (e.g. data.info.labelled)")

    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Run prediction on new data")
    predict_parser.add_argument("input_file", type=str, help="Input file name (e.g. testdata.json.gz)")

    args = parser.parse_args()

    # Execute command
    if args.command == "train":
        try:
            train_bagged_models(args.dataset, args.label_file)
        except Exception as e:
            print(f"Error during training: {e}")
            sys.exit(1)

    elif args.command == "predict":
        data_dir = "data"
        model_dir = "models"
        num_bags = 5

        file_path = os.path.join(data_dir, args.input_file)
        if not os.path.exists(file_path):
            print(f"Error: {args.input_file} not found in '{data_dir}' folder.")
            sys.exit(1)

        try:
            run_prediction(
                input_file=args.input_file,
                model_dir=model_dir,
                data_dir=data_dir,
                num_bags=num_bags,
            )
        except Exception as e:
            print(f"Error during prediction: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()