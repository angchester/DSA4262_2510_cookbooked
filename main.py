import argparse
from src.train.train_model import train_bagged_models
from src.predict.task1_predict import run_task1_prediction

def main():
    p = argparse.ArgumentParser(description="RNA Model Project")
    p.add_argument("mode", choices=["train", "predict"])
    p.add_argument("--num_bags", type=int, default=5)
    args = p.parse_args()

    if args.mode == "train":
        train_bagged_models(
            data_path="data/dataset0_test.json.gz",
            label_path="data/data.info_test.labelled",
            save_dir="models",
            num_bags=args.num_bags,
        )
    elif args.mode == "predict":
        run_task1_prediction(model_dir="models", data_dir="data", num_bags=args.num_bags)
    else:
        print("Invalid choice, please choose train or predict")
        
if __name__ == "__main__":
    main()
