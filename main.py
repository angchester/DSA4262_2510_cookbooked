import os
import sys
from src.prediction import run_prediction

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 main.py <filename.json or filename.json.gz>")
        sys.exit(1)

    input_file = sys.argv[1]
    data_dir = "data"
    model_dir = "models"
    num_bags = 5

    file_path = os.path.join(data_dir, input_file)
    if not os.path.exists(file_path):
        print(f"Error: {input_file} not found in '{data_dir}' folder.")
        sys.exit(1)

    run_prediction(
        input_file=input_file,
        model_dir=model_dir,
        data_dir=data_dir,
        num_bags=num_bags
    )

if __name__ == "__main__":
    main()
