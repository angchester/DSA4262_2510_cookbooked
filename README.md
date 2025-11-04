# Genomics Project: Prediction of m6A RNA modifications from direct RNA-Seq data

This repository contains the implementation of a machine learning approach developed by Team cookbooked to identify m6A RNA modifications from direct RNA sequencing data.

## Folder Structure

```
DSA4262_2510_cookbooked/
│
├── main.py                        # Entry point
├── requirements.txt               # Dependencies
│
├── data/                          # Input datasets (Sample test + SGNex data)
│   └── testdata.json.gz           # Sample test data
│        
├── models/                        # Trained bagged models (.pt)
│   ├── bag1_finalmodel.pt
│   ├── bag2_finalmodel.pt
│   └── …
│
├── predictions/                   # Output folder (auto-created upon running predictions)
│  
└── src/
    ├── data_utils.py              # Data loading & pre-processing
    ├── model_arch.py              # Model + FocalLoss 
    ├── prediction.py              # To run predictions
    └── train_model                # Training script for final model
```

# Setting up
Do follow these steps to ensure that the model can run smoothly
**Machine Setup**
1) For AWS users:
Start a new instance. We recommend using a machine with at least `m4.4xlarge` specifications and `50 GB SSD` storage.
2) For local users:
Ensure that you have Python Version $\geq$ 3.10 installed.

**_Cloning Repository_**

To clone the repository to your device, run

```bash
git clone https://github.com/angchester/DSA4262_2510_cookbooked.git
```
If cloned successfully, the folder `DSA4262_2510_cookbooked` will be created. 

Ensure that you are in the `DSA4262_2510_cookbooked` folder before proceeding.
```bash
cd DSA4262_2510_cookbooked
```

**_Install Software/Packages_**
1) Installing pip
   
If you are using an AWS instance, run the following command to install pip and essential packages
```bash
sudo apt update && sudo apt upgrade -y && sudo apt install -y git python3 python3-pip python3-venv
```
If you are on AWS instance, a few pop-ups might appear. Select Ok to continue.

2) Installing required packages
   
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
Once all packages are installed, you are ready to run the model.

# Running the model

1) Preparing Input data

All dataset files must be placed inside the `/data` directory.
For example, to test the model with the provided sample dataset, ensure the file is located at `/data/testdata.json.gz`

2) Running Predictions

Once your input data is in `/data`, you can run the prediction script using `main.py`.
The script accepts data files in either `.json.gz` or `.json` format (If you are using Windows, replace python3 with python)

```bash
python3 main.py predict <data_file>.json.gz
# or
python3 main.py predict <data_file>.json
```

Replace `<data_file>` with the filename only (e.g `testdata`)

**Note**: You **do not need to include the folder path** such as `/data/testdata.json.gz`. The script automatically searches for all files inside the `/data` directory, so you only need to specify the filename (e.g `testdata.json.gz`)

To test the model using the provided sample data
```bash
python3 main.py predict testdata.json.gz
```

**Note**: Our model is an ensemble of 5 bagged models, so the prediction script will run 5 separate inference rounds (one per bag) before averaging the results to produce the final prediction. This process may take some time to complete, depending on your hardware performance and the size of the data. Please allow the script to finish without interruption.

3) Prediction Output location

After a successful prediction run, the results will be saved automatically in `/predictions` folder as a `.csv` file. You may find the predicted sample data at `/predictions/testdata_pred.csv`

**_Predicting on SGNex data_**

To access the SGNex S3 bucket, install AWS CLI
```bash
sudo apt install awscli
```
You might be prompted to continue, just enter 'Y'

If you are on AWS instance, a few pop-ups might appear. Select Ok to continue.

To download a specific dataset folder from SGNex, you can run
```bash
aws s3 cp --no-sign-request \
s3://sg-nex-data/data/processed_data/m6Anet/<FOLDER_NAME>/data.json \
data/<FOLDER_NAME>.json
```
Replace `<FOLDER_NAME>` with the specific dataset folder name. The first path is the remote S3 location while the second path save the data into the local `/data` folder.

**_Example of predicting SGNex data_**

For example, if we wish to predict `SGNex_A549_directRNA_replicate5_run1`, run the following command
```bash
aws s3 cp --no-sign-request \
s3://sg-nex-data/data/processed_data/m6Anet/SGNex_A549_directRNA_replicate5_run1/data.json \
data/SGNex_A549_directRNA_replicate5_run1.json
```
This saves `SGNex_A549_directRNA_replicate5_run1.json` in `/data`. Run the following command to do prediction
```bash
python3 main.py predict SGNex_A549_directRNA_replicate5_run1.json
```

Upon successful completion, the predictions will be saved as `/predictions/SGNex_A549_directRNA_replicate5_run1_pred.csv`

# Team Members
1) ANG QI WEI CHESTER
2) CHIONH WAN SIM
3) NG JUNLIN BENJAMIN
4) SIM WEI





