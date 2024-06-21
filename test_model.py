import pandas as pd
from joblib import load
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

class CreditRiskDataLoader:
    def __init__(self, x_data_path, y_data_path):
        self.x_data_path = x_data_path
        self.y_data_path = y_data_path
        self.X = None
        self.y = None

    def load_data(self):
        self.X = pd.read_csv(self.x_data_path)
        self.y = pd.read_csv(self.y_data_path)
        return self.X, self.y
    
    def preprocess_data_from(self, path):
        data = pd.read_csv(path)
        self.X = data
        self.X = self.X.drop_duplicates()
        self.X = self.X.dropna()
        self.X = pd.get_dummies(self.X, columns=['person_home_ownership', 'loan_intent'])
        mapping = {
            'A': 0, 
            'B': 1, 
            'C': 2, 
            'D': 3, 
            'E': 4, 
            'F': 5, 
            'G': 6, 
            }
        self.X['loan_grade'] = self.X['loan_grade'].map(mapping)
        mapping1 = {
            'N': 0, 
            'Y': 1, 
            }
        self.X['cb_person_default_on_file'] = self.X['cb_person_default_on_file'].map(mapping1)        
        self.X = self.X[self.X['person_age'] < 100]
        self.y = self.X['loan_status']
        self.X.drop(['loan_status'], axis=1, inplace=True)
        self.X.info()
        
        return self.X, self.y

if __name__ == '__main__':
    x_data_path = "data/test_dataset.csv"
    y_data_path = "data/test_dataset_label.csv"
    data_path = "data/credit_risk_dataset.csv"
    models = [
        #'models/RFC_model.joblib',
        #'models/RFC_under_model.joblib',
        #'models/XGB_under_model.joblib',
        'models/XGB_model.joblib'
    ]

    data_loader = CreditRiskDataLoader(x_data_path, y_data_path)
    #X_test, y_test = data_loader.load_data()
    X_test, y_test = data_loader.preprocess_data_from(data_path)

    for model_path in models:
        model = load(model_path)
        
        y_pred = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_prob)
        cm = confusion_matrix(y_test, y_pred)
        clf_report = classification_report(y_test, y_pred)
        
        print(f"Evaluation results for {model_path}:")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")
        print(f"ROC AUC: {roc_auc}")
        print(f"Confusion Matrix:\n{cm}")
        print(f"Classification Report:\n{clf_report}")
        print()

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['Not Default', 'Default'], yticklabels=['Not Default', 'Default'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix: {model_path.split("/")[-1]}')
        plt.show()

    for model_path in models:
    
        model = load(model_path)
        for index, row in X_test.iterrows():
            X_single_sample = pd.DataFrame([row])
        
            y_pred = model.predict(X_single_sample)
        
            print("Results for:")
            #print(f"{row}")
            print(f"Prediction: {y_pred}")