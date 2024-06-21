## Requirements:    
-Python 3.x    
-Necessary Python packages (install using pip install -r requirements.txt):        
- 'pandas'    
- 'joblib'    
- 'scikit-learn'    
    <li>seaborn    
    <li>matplotlib    
    <li>xgboost    
    <li>imbalanced-learn    
    <li>scipy 

## Installing packages
```console
pip install -r requirements.txt
```

## Data Files    
The scripts requires the following data files:    
    <li>data/test_dataset.csv: The feature set for the test data(encoded).        
    <li>data/test_dataset_label.csv: The labels for the test data(encoded).            
    <li>data/credit_risk_dataset.csv: The dataset for learning(non encoded).   
    <li>models/*.joblib: Pre-trained models saved in the specified directory.    


## How to Use    
    <li>Data and models can also be preparred using start_learing.py or by running all cells in projekt.ipynd    
    <li>Prepare the Data: Ensure you have the test dataset and its labels in the data directory.    
    <li>Place the Models: Save your trained models in the models directory.    
    <li>Run the Script: Execute the script to evaluate the models and visualize the results.    
    <li>Script can also be run in test_m.ipynb    


## Teaching models
```console
python model_learning.py
```
or running all cells in projekt.ipynd


## Using models
```console
python test_model.py
```
or running all cells in test_m.ipynd


CreditRiskDataLoader Class:    
<li>__init__: Initializes paths for test feature data and labels.    
<li>load_data: Loads the test data and labels from CSV files.    
<li>preprocess_data_from: (Optional) Preprocesses the data from a specified CSV file.    


CreditRiskDataLoader Class:    
<li>__init__: Initializes the path to the data, categorical features, numerical features, and the target label column.    
<li>load_data: Loads the data, removes duplicates and missing values, separates the labels from the features.    
<li>preprocess_data: Processes the data by encoding categorical variables and mapping values.    
<li>get_train_test_split: Splits the data into training and test sets and saves the test set to CSV files.    


CreditRiskClassifier Class:    
<li>__init__: Initializes the classifier type.    
<li>build_pipeline: Builds a pipeline for the chosen classifier, with optional random sampling.    
<li>train_model: Trains the model using RandomizedSearchCV for hyperparameter optimization.    
<li>evaluate_model: Evaluates the model on test data, returning various performance metrics.    
<li>save_model: Saves the trained model to a file.    
