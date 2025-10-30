# Genomics Project: Prediction of m6A RNA modifications from direct RNA-Seq data

This repository contains the implementation of a machine learning approach developed by Team cookbooked to identify m6A RNA modifications from direct RNA sequencing data.

## Folder Structure

```
DSA4262_2510_cookbooked/
│
├── main.py                        # Entry point
├── requirements.txt               # Dependencies
│
├── data/                          # Input datasets 
│   └── testdata.json.gz           # Test data to try out
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
    └── prediction.py              # To run predictions
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
If cloned successfully, you should see folder `DSA4262_2510_cookbooked` created.

**_Install Software/Packages_**
1) Installing pip
   
If you are using an AWS instance, run the following command to install pip and essential packages
```bash
sudo apt update && sudo apt upgrade -y && sudo apt install -y git python3 python3-pip python3-venv
```
A pop-up might appear in AWS instance, select Ok to continue.

2) Installing required packages
   
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
Once all packages are installed, you are ready to run the model.

# Running the model

Ensure that you are in the `DSA4262_2510_cookbooked` folder before running any commands.

To run the prediction script, you can run main.py and input in the data in `.json.gz` or `.json` in the following format. (If you are using Windows, replace python3 with python)
```bash
python3 main.py {data}.json.gz
# or
python3 main.py {data}.json
```
Replace `{data}` with the name of the dataset file.

To test the model with a sample data
```bash
python3 main.py testdata.json.gz
```
**Note** that our model is an ensemble of 5 bagged models, so the prediction script will run 5 inference rounds (one per bag) before averaging the results. This process may take some time to complete, depending on your hardware performance and the size of the data. Please allow the script to finish without interruption.

Upon successful completion, the predictions will be saved as a `.csv` file inside the `/predictions` folder.

**_Predicting on SGNex data_**

To access the SGNex S3 bucket, install AWS CLI
```bash
sudo apt install awscli
```
You might be prompted to continue, just enter 'Y'
A pop-up might appear in AWS instance, select Ok to continue.

To download a specific dataset folder from SGNex, you can run
```bash
aws s3 cp --no-sign-request \
s3://sg-nex-data/data/processed_data/m6Anet/<FOLDER_NAME>/data.json \
data/<FOLDER_NAME>.json
```
Replace `<FOLDER_NAME>` with the specific dataset folder name. The first path is the remote S3 location while the second path save the data into the local `/data` folder.

**_Example of predicting SGNex data_**

For example, if we wish to predict `SGNex_A549_directRNA_replicate5_run1`, we can run the following commands
```bash
aws s3 cp --no-sign-request \
s3://sg-nex-data/data/processed_data/m6Anet/SGNex_A549_directRNA_replicate5_run1/data.json \
data/SGNex_A549_directRNA_replicate5_run1.json
```
'SGNex_A549_directRNA_replicate5_run1.json' should appear in `/data`. To run the prediction on this,
```bash
python3 main.py SGNex_A549_directRNA_replicate5_run1.json
```

# Team Members
1) ANG QI WEI CHESTER
2) CHIONH WAN SIM
3) NG JUNLIN BENJAMIN
4) SIM WEI





