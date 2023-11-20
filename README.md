## Fall2023_IDS706 Mini Project 12: Use MLflow to Manage an ML Project
### by Jiayi Zhou [![CI](https://github.com/nogibjj/Fall2023_IDS706_MiniProject12_JiayiZhou/actions/workflows/cicd.yml/badge.svg)](https://github.com/nogibjj/Fall2023_IDS706_MiniProject12_JiayiZhou/actions/workflows/cicd.yml)

### Purpose
This is for class data engineering mini project 12. It creates a simple machine-learning model--linear regression and uses MLflow to manage the project, including tracking metrics.

### Steps:
1. Data Loading and Preprocessing:
2. Load the dataset into a Pandas DataFrame.
3. Extract relevant features (e.g., 'team1_win', 'tie').
4. Model Training:
    -Split the data into training and testing sets.
    -Train the model on the training set.
5. MLflow Integration:
    1. Use MLflow to log parameters, metrics, and the trained model.
    2. Log parameters such as the model type and data path.
    3. Log metrics like accuracy.

### Preparation: 
1. git clone the repo
2. install: `make install`
3. run: `python main.py`
4. run: `mlflow ui` to view mlflow interface 
5. view saved model and other artifacts in `mlruns/0`

### Check Format and Test Errors: 
1. Format code `make format`
2. Lint code `make lint`
3. Test coce `make test`


