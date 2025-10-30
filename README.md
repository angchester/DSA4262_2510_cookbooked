# Genomics Project: Prediction of m6A RNA modifications from direct RNA-Seq data

This GitHub Repo contains the codes for a machine learning method to identify m6A RNA modifications from direct RNA sequencing data, done by team cookbooked.

## Folder Structure

```
For-Github/
│
├── main.py                        # Entry point (Train + Predict)
├── requirements.txt               # Dependencies
│
├── data/                          # Input datasets 
│   ├── dataset0_test.json.gz      # Labelled training data (10% subset of dataset0)
│   ├── dataset1_test.json.gz      # Optional unlabelled data (10% subset of dataset1)
│   └── data.info_test.labelled    # Label file for dataset0 (10% subset of dataset0)
│
├── models/                        # Trained bagged models (.pt)
│   ├── bag1_finalmodel.pt
│   ├── bag2_finalmodel.pt
│   └── …
│
├── predictions/                   # Outputs from Task 1 prediction
│   ├── task1_dataset0_pred.csv
│   ├── task1_dataset1_pred.csv
│   ├── task1_dataset0_metrics.json
│   ├── dataset0_confusion_matrix.png
│   ├── dataset0_aucroc.png
│   └── dataset0_aucpr.png
│
└── src/
    ├── data_utils.py              # Data loading & pre-processing
    ├── model_arch.py              # Model + FocalLoss
    ├── train/    
    │   └── model_train.py         # Bagged training (Task 1)
    └── predict/
        └── task1_predict.py       # Ensemble prediction (Task 1)
```

# Setting up

## Clone Repository

To clone the repository into your device, you can run the following command

```bash
git https://github.com/angchester/DSA4262_2510_cookbooked.git
```




To enter the script
  python main.py train
  python main.py predict


WARNING: DO NOT RUN main.py train, as the model takes ~12 hours to fully train. The .pt files of the full model is provided in the models folder, which will allow you to run the
script to do prediction for task 1 and task 2.

