import argparse
from src.train.train_model import train_bagged_models
from src.predict.task1_predict import run_task1_prediction
from src.predict.task2_predict import run_task2_prediction

def main():
    p = argparse.ArgumentParser(description="RNA Model Project")
    p.add_argument("mode", choices=["train", "task1", "task2"])
    p.add_argument("--num_bags", type=int, default=5)
    args = p.parse_args()

    if args.mode == "train":
        train_bagged_models(
            data_path="data/dataset0.json.gz",
            label_path="data/data.info.labelled",
            save_dir="models",
            num_bags=args.num_bags,
        )
    elif args.mode == "task1":
        run_task1_prediction(model_dir="models", data_dir="data", num_bags=args.num_bags)
    elif args.mode == "task2":
        run_task2_prediction(model_dir="models", task2_root="task2_data", num_bags=args.num_bags)

if __name__ == "__main__":
    main()
