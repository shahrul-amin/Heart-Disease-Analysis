"""
Automated Pipeline for Heart Disease Prediction
Tests all combinations of preprocessing methods and models
"""

import pandas as pd
import numpy as np
import json
import itertools
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

class AutomatedPipeline:
    """
    Automated pipeline that tests all combinations of:
    - Null handling methods: intelligent imputation, drop rows
    - Outlier handling: cap, remove  
    - Categorical encoding: one-hot, label encoding
    - Models: SVM, XGBoost, Random Forest
    """
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.original_data = None
        self.results = []
        self.best_result = None
        
        # Define preprocessing combinations
        self.null_methods = ['intelligent', 'drop_rows']
        self.outlier_methods = ['cap', 'remove']
        self.encoding_methods = ['onehot', 'label']
        self.models = ['SVM', 'XGB', 'RandomForest']
        
    def load_data(self):
        """Load the dataset"""
        try:
            self.original_data = pd.read_csv(self.data_path, index_col='id')
            print(f"Data loaded successfully. Shape: {self.original_data.shape}")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def handle_nulls(self, df, method):
        """Handle null values based on specified method"""
        df_processed = df.copy()
        
        if method == 'intelligent':
            # Intelligent imputation based on data type
            for col in df_processed.columns:
                if df_processed[col].isnull().any():
                    if df_processed[col].dtype in ['int64', 'float64']:
                        # For numeric: use median for skewed, mean for normal
                        if abs(df_processed[col].skew()) > 1:
                            df_processed[col].fillna(df_processed[col].median(), inplace=True)
                        else:
                            df_processed[col].fillna(df_processed[col].mean(), inplace=True)
                    else:
                        # For categorical: use mode
                        df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
        
        elif method == 'drop_rows':
            # Drop rows with any null values
            df_processed = df_processed.dropna()
        
        return df_processed
    
    def handle_outliers(self, df, method):
        """Handle outliers based on specified method"""
        df_processed = df.copy()
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col == 'num':  # Skip target variable
                continue
                
            Q1 = df_processed[col].quantile(0.25)
            Q3 = df_processed[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            if method == 'cap':
                # Cap outliers to bounds
                df_processed[col] = df_processed[col].clip(lower=lower_bound, upper=upper_bound)
            
            elif method == 'remove':
                # Remove outliers
                mask = (df_processed[col] >= lower_bound) & (df_processed[col] <= upper_bound)
                df_processed = df_processed[mask]
        
        return df_processed
    
    def encode_categorical(self, df, method):
        """Encode categorical variables based on specified method"""
        df_processed = df.copy()
        
        # Convert boolean strings to numeric first
        boolean_cols = []
        for col in df_processed.columns:
            if df_processed[col].dtype == 'object':
                unique_vals = df_processed[col].dropna().unique()
                if len(unique_vals) == 2 and set(str(v).upper() for v in unique_vals).issubset({'TRUE', 'FALSE'}):
                    df_processed[col] = df_processed[col].map({'TRUE': 1, 'False': 0, 'True': 1, 'FALSE': 0})
                    boolean_cols.append(col)
        
        # Get categorical columns (excluding converted boolean and target)
        categorical_cols = []
        for col in df_processed.columns:
            if col not in boolean_cols and col != 'num' and df_processed[col].dtype == 'object':
                categorical_cols.append(col)
        
        if method == 'onehot' and categorical_cols:
            # One-hot encoding
            df_processed = pd.get_dummies(df_processed, columns=categorical_cols, prefix=categorical_cols)
        
        elif method == 'label' and categorical_cols:
            # Label encoding
            for col in categorical_cols:
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col].astype(str))
        
        return df_processed
    
    def prepare_features_target(self, df):
        """Prepare features and target variables"""
        # Ensure target is binary
        if 'num' in df.columns:
            y = (df['num'] > 0).astype(int)  # Convert to binary classification
            X = df.drop('num', axis=1)
        else:
            raise ValueError("Target column 'num' not found")
        
        # Ensure all features are numeric
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
        
        return X, y
    
    def train_model(self, X_train, X_test, y_train, y_test, model_name):
        """Train specified model and return metrics"""
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Initialize model
        if model_name == 'SVM':
            model = SVC(random_state=42, probability=True)
        elif model_name == 'XGB':
            model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        elif model_name == 'RandomForest':
            model = RandomForestClassifier(random_state=42, n_estimators=100)
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Calculate sensitivity (recall for positive class)
        sensitivity = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        
        return {            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'sensitivity': sensitivity,
            'model': model
        }
    
    def run_pipeline(self):
        """Run the complete automated pipeline"""
        if not self.load_data():
            return None
        
        # Generate all combinations
        combinations = list(itertools.product(
            self.null_methods,
            self.outlier_methods, 
            self.encoding_methods,
            self.models
        ))
        
        total_combinations = len(combinations)
        expected_combinations = len(self.null_methods) * len(self.outlier_methods) * len(self.encoding_methods) * len(self.models)
        
        print(f"Expected combinations: {expected_combinations}")
        print(f"Generated combinations: {total_combinations}")
        print(f"Methods: nulls={len(self.null_methods)}, outliers={len(self.outlier_methods)}, encoding={len(self.encoding_methods)}, models={len(self.models)}")
        print(f"Testing {total_combinations} combinations...")
        
        # Verify we have the right number of combinations
        assert total_combinations == 24, f"Expected 24 combinations, got {total_combinations}"
        
        for i, (null_method, outlier_method, encoding_method, model_name) in enumerate(combinations, 1):
            try:
                print(f"Processing combination {i}/{total_combinations}: "
                      f"Nulls={null_method}, Outliers={outlier_method}, "
                      f"Encoding={encoding_method}, Model={model_name}")
                
                # Start with fresh data
                df_processed = self.original_data.copy()
                
                # Apply preprocessing steps
                df_processed = self.handle_nulls(df_processed, null_method)
                df_processed = self.handle_outliers(df_processed, outlier_method)
                df_processed = self.encode_categorical(df_processed, encoding_method)
                
                # Prepare features and target
                X, y = self.prepare_features_target(df_processed)
                  # Skip if too few samples (reduced threshold)
                if len(X) < 20:
                    print(f"  Skipping - insufficient samples after preprocessing: {len(X)}")
                    # Still record this as a failed combination
                    result = {
                        'combination_id': i,
                        'null_handling': null_method,
                        'outlier_handling': outlier_method,
                        'categorical_encoding': encoding_method,
                        'model': model_name,
                        'sample_size': len(X),
                        'features_count': 0,
                        'accuracy': 0,
                        'precision': 0,
                        'recall': 0,
                        'f1_score': 0,
                        'sensitivity': 0,
                        'timestamp': datetime.now().isoformat(),
                        'status': 'failed_insufficient_data'
                    }
                    self.results.append(result)
                    continue
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                
                # Train model and get metrics
                metrics = self.train_model(X_train, X_test, y_train, y_test, model_name)
                  # Store results
                result = {
                    'combination_id': i,
                    'null_handling': null_method,
                    'outlier_handling': outlier_method,
                    'categorical_encoding': encoding_method,
                    'model': model_name,
                    'sample_size': len(X),
                    'features_count': X.shape[1],
                    'accuracy': metrics['accuracy'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1_score': metrics['f1_score'],
                    'sensitivity': metrics['sensitivity'],
                    'timestamp': datetime.now().isoformat(),                    'status': 'success'
                }
                
                self.results.append(result)
                print(f"  Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")
                
            except Exception as e:
                print(f"  Error in combination {i}: {e}")
                # Record failed combination
                result = {
                    'combination_id': i,
                    'null_handling': null_method,
                    'outlier_handling': outlier_method,
                    'categorical_encoding': encoding_method,
                    'model': model_name,
                    'sample_size': 0,
                    'features_count': 0,
                    'accuracy': 0,
                    'precision': 0,
                    'recall': 0,
                    'f1_score': 0,
                    'sensitivity': 0,
                    'timestamp': datetime.now().isoformat(),
                    'status': f'failed_error: {str(e)}'
                }
                self.results.append(result)
                continue
        
        # Find best result
        if self.results:
            self.best_result = max(self.results, key=lambda x: x['f1_score'])
            print(f"\nBest result: F1={self.best_result['f1_score']:.4f} "
                  f"(Combination {self.best_result['combination_id']})")
        
        return self.results
    
    def save_results(self, filename='pipeline_results.json'):
        """Save results to JSON file"""
        if not self.results:
            print("No results to save")
            return
        
        output = {
            'metadata': {
                'total_combinations': len(self.results),
                'best_combination': self.best_result,
                'run_timestamp': datetime.now().isoformat()
            },
            'results': self.results
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"Results saved to {filename}")
        return filename
    
    def get_summary_stats(self):
        """Get summary statistics of all results"""
        if not self.results:
            return None
        
        df_results = pd.DataFrame(self.results)
        
        summary = {
            'total_combinations': len(self.results),
            'best_f1_score': df_results['f1_score'].max(),
            'average_f1_score': df_results['f1_score'].mean(),
            'best_accuracy': df_results['accuracy'].max(),
            'average_accuracy': df_results['accuracy'].mean(),
            'method_performance': {
                'by_null_method': df_results.groupby('null_handling')['f1_score'].mean().to_dict(),
                'by_outlier_method': df_results.groupby('outlier_handling')['f1_score'].mean().to_dict(),
                'by_encoding_method': df_results.groupby('categorical_encoding')['f1_score'].mean().to_dict(),
                'by_model': df_results.groupby('model')['f1_score'].mean().to_dict()
            }
        }
        
        return summary

if __name__ == "__main__":
    # Example usage
    pipeline = AutomatedPipeline("data/heart_disease_uci.csv")
    results = pipeline.run_pipeline()
    
    if results:
        pipeline.save_results()
        summary = pipeline.get_summary_stats()
        print("\nSummary Statistics:")
        print(json.dumps(summary, indent=2))
