import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from joblib import dump
from scipy.stats import randint, uniform


class CreditRiskDataLoader:
    def __init__(self, data_path, categorical_features, numeric_features, target_column):
        self.data_path = data_path
        self.categorical_features = categorical_features
        self.numeric_features = numeric_features
        self.target_column = target_column
        self.data = None
        self.label = None
        
    def load_data(self):
        self.data = pd.read_csv(self.data_path)
        
        self.data = self.data.drop_duplicates()
        self.data = self.data.dropna()
        self.data = self.data[self.data['person_age'] < 100]

        self.label = self.data[self.target_column]
        self.data.drop([self.target_column], axis=1, inplace=True)
        
        return self.data, self.label
    
    def preprocess_data(self): 
        self.data = self.data.drop_duplicates()
        self.data = self.data.dropna()
        self.data = pd.get_dummies(self.data, columns=['person_home_ownership', 'loan_intent'])
        mapping = {
            'A': 0, 
            'B': 1, 
            'C': 2, 
            'D': 3, 
            'E': 4, 
            'F': 5, 
            'G': 6, 
            }
        self.data['loan_grade'] = self.data['loan_grade'].map(mapping)
        mapping1 = {
            'N': 0, 
            'Y': 1, 
            }
        self.data['cb_person_default_on_file'] = self.data['cb_person_default_on_file'].map(mapping1)
        self.data.info()
        
        return self.data
    
    def get_train_test_split(self, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(self.data, self.label, test_size=test_size, random_state=random_state)
        X_test.to_csv('data/test_dataset.csv', index=False)
        y_test.to_csv('data/test_dataset_label.csv', index=False)
        return X_train, X_test, y_train, y_test
    
class CreditRiskClassifier:
    def __init__(self, classifier_type='RFC'):
        self.classifier_type = classifier_type
        self.pipeline = None
        self.grid_search = None
        self.best_model = None
        
    def build_pipeline(self):
        if self.classifier_type == 'RFC':
            self.pipeline = Pipeline(steps=[
                ('preprocessing', StandardScaler()),
                ('classifier', RandomForestClassifier(random_state=42))
            ])
        elif self.classifier_type == 'XGB':
            self.pipeline = Pipeline(steps=[
                ('preprocessing', StandardScaler()),
                ('classifier', XGBClassifier(random_state=42))
            ])
        elif self.classifier_type == 'RFC_under':
            self.pipeline = ImbPipeline(steps=[
                ('preprocessing', StandardScaler()),
                ('sampling', RandomUnderSampler()), 
                ('classifier', RandomForestClassifier(random_state=42))
            ])
        elif self.classifier_type == 'XGB_under':
            self.pipeline = ImbPipeline(steps=[
                ('preprocessing', StandardScaler()),
                ('sampling', RandomUnderSampler()), 
                ('classifier', XGBClassifier(random_state=42))
            ])
        else:
            raise ValueError(f"Unsupported classifier type: {self.classifier_type}")
        
    def train_model(self, X_train, y_train, param_grid, cv=5, scoring='roc_auc', n_iter=100, verbose=10, random_state=42):
        self.grid_search = RandomizedSearchCV(
            estimator=self.pipeline, 
            param_distributions=param_grid, 
            n_iter=n_iter, 
            cv=StratifiedKFold(n_splits=cv, random_state=123, shuffle=True), 
            scoring=scoring, 
            random_state=random_state, 
            verbose=verbose
            )
        self.grid_search.fit(X_train, y_train)
        
        self.best_model = self.grid_search.best_estimator_
        
    def evaluate_model(self, X_test, y_test):
        y_pred = self.best_model.predict(X_test)
        y_pred_prob = self.best_model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_prob)
        
        return accuracy, precision, recall, f1, roc_auc

    def save_model(self):
        if self.best_model:
            dump(self.best_model, f"models/{self.classifier_type}_model.jolib")
            print("Model saved successfully.")
        else:
            print("No model to save. Please train a model first.")

if __name__ == '__main__':
    data_path = "data/credit_risk_dataset.csv"
    categorical_features = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
    numeric_features = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']
    target_column = 'loan_status'
    
    data_loader = CreditRiskDataLoader(data_path, categorical_features, numeric_features, target_column)
    
    data_loader.load_data()
    preprocessed_data = data_loader.preprocess_data()
    
    X_train, X_test, y_train, y_test = data_loader.get_train_test_split()

    classifiers = ['XGB'] #,'RFC', 'RFC_under', 'XGB_under']
    param_grid_XGB = {
            'classifier__learning_rate': uniform(0.01, 0.5),
            'classifier__max_depth': randint(3, 10),
            'classifier__n_estimators': randint(100, 1000),
            'classifier__min_child_weight': uniform(1, 10),
            'classifier__subsample': uniform(0.6, 0.4),
            'classifier__colsample_bytree': uniform(0.6, 0.4),
            'classifier__gamma': uniform(0, 0.5)
        }

    param_dist_RFC = {
            'classifier__n_estimators': randint(100, 500),
            'classifier__max_depth': [None] + list(randint(10, 50).rvs(10)),
            'classifier__min_samples_split': randint(2, 20), 
            'classifier__min_samples_leaf': randint(1, 4),
            'classifier__max_samples': uniform(0.1, 0.9),
            'classifier__max_leaf_nodes': [None] + list(randint(10, 30).rvs(10)),
            'classifier__criterion': ['gini', 'entropy'],
            'classifier__class_weight': [None, 'balanced', 'balanced_subsample']
        }
    
    for classifier_type in classifiers:
        classifier = CreditRiskClassifier(classifier_type=classifier_type)
        
        classifier.build_pipeline()

        if classifier_type in ['RFC','RFC_under']:
            param_grid = param_dist_RFC
        else:
            param_grid = param_grid_XGB
        
        classifier.train_model(X_train, y_train, param_grid)
        classifier.save_model()
        
        accuracy, precision, recall, f1, roc_auc = classifier.evaluate_model(X_train, y_train)
        print("On train dataset")
        print(f"Classifier: {classifier_type}")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")
        print(f"ROC AUC: {roc_auc}")
        print()

        accuracy, precision, recall, f1, roc_auc = classifier.evaluate_model(X_test, y_test)
        print("On test dataset")
        print(f"Classifier: {classifier_type}")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")
        print(f"ROC AUC: {roc_auc}")
        print()
