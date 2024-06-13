Requirements:
-Python 3.x
-Necessary Python packages (install using pip install -r requirements.txt):
    -pandas
    -joblib
    -scikit-learn
    -seaborn
    -matplotlib
    -xgboost
    -imbalanced-learn
    -scipy

Data Files
The script requires the following data files:
    -data/test_dataset.csv: The feature set for the test data(encoded).
    -data/test_dataset_label.csv: The labels for the test data(encoded).
    -models/*.joblib: Pre-trained models saved in the specified directory(non encoded).

How to Use
    -Data and models can also be preparred using start_learing.py or by running all cells in projekt.ipynd
    -Prepare the Data: Ensure you have the test dataset and its labels in the data directory.
    -Place the Models: Save your trained models in the models directory.
    -Run the Script: Execute the script to evaluate the models and visualize the results.
    -Script can also be run in test_m.ipynb


