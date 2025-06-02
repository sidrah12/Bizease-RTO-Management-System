import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class RTOPredictor:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self.model_type = None
        self.model_path = 'models/rto_model.joblib'
        self.feature_importance = None
        self.training_history = []
        
        # Define feature groups
        self.categorical_features = [
            'address_classification',
            'status',
            'payment_method',
            'city',
            'state'
        ]
        
        self.numerical_features = [
            'quantity',
            'pincode'  # Treated as numerical for distance calculation
        ]
        
        self.all_features = self.categorical_features + self.numerical_features
        
    def _create_preprocessor(self):
        """Create a preprocessing pipeline for the data"""
        # Create preprocessing steps
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        # Combine preprocessing steps
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, self.categorical_features),
                ('num', numerical_transformer, self.numerical_features)
            ],
            remainder='drop'
        )
        
        return self.preprocessor
    
    def _create_model_pipeline(self, model_type='random_forest'):
        """Create a model pipeline with preprocessing"""
        if self.preprocessor is None:
            self._create_preprocessor()
            
        if model_type == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced'
            )
            self.model_type = 'random_forest'
        elif model_type == 'gradient_boosting':
            model = GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
            self.model_type = 'gradient_boosting'
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        # Create pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('classifier', model)
        ])
        
        self.model = pipeline
        return pipeline
    
    def _extract_feature_names(self, X):
        """Extract feature names after preprocessing"""
        # Get feature names from the preprocessor
        cat_features = self.preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(self.categorical_features)
        num_features = self.numerical_features
        
        self.feature_names = np.concatenate([cat_features, num_features])
        return self.feature_names
    
    def train(self, training_data, target_column='rto_risk', model_type='random_forest', tune_hyperparameters=False):
        """Train the RTO prediction model with optional hyperparameter tuning"""
        # Convert to DataFrame if not already
        if not isinstance(training_data, pd.DataFrame):
            training_data = pd.DataFrame(training_data)
            
        # Prepare features and target
        X = training_data[self.all_features]
        y = training_data[target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Create model pipeline
        self._create_model_pipeline(model_type)
        
        # Hyperparameter tuning if requested
        if tune_hyperparameters:
            print("Tuning hyperparameters...")
            if self.model_type == 'random_forest':
                param_grid = {
                    'classifier__n_estimators': [100, 200, 300],
                    'classifier__max_depth': [10, 15, 20, None],
                    'classifier__min_samples_split': [2, 5, 10],
                    'classifier__min_samples_leaf': [1, 2, 4]
                }
            else:  # gradient_boosting
                param_grid = {
                    'classifier__n_estimators': [100, 200, 300],
                    'classifier__learning_rate': [0.01, 0.1, 0.2],
                    'classifier__max_depth': [3, 5, 7]
                }
                
            grid_search = GridSearchCV(
                self.model, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
        else:
            # Train the model
            self.model.fit(X_train, y_train)
        
        # Extract feature names
        self._extract_feature_names(X_train)
        
        # Calculate feature importance
        if self.model_type == 'random_forest':
            self.feature_importance = self.model.named_steps['classifier'].feature_importances_
        elif self.model_type == 'gradient_boosting':
            self.feature_importance = self.model.named_steps['classifier'].feature_importances_
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        
        # Store training history
        self.training_history.append({
            'timestamp': datetime.now(),
            'model_type': self.model_type,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'n_samples': len(X_train) + len(X_test)
        })
        
        # Print evaluation metrics
        print(f"Model Evaluation:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Print confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        return accuracy, precision, recall, f1
    
    def predict(self, data):
        """Predict RTO risk for new orders"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # Convert to DataFrame if not already
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
            
        # Handle missing values
        data = data.fillna('Unknown')
            
        # Ensure all required features are present
        for feature in self.all_features:
            if feature not in data.columns:
                data[feature] = 'Unknown'  # Default value for missing features
                
        # Convert numeric features
        for feature in self.numerical_features:
            data[feature] = pd.to_numeric(data[feature], errors='coerce').fillna(0)
        
        # Make predictions
        predictions = self.model.predict(data)
        probabilities = self.model.predict_proba(data)
        
        return predictions, probabilities
    
    def get_feature_importance(self, top_n=10):
        """Get the top N most important features"""
        if self.feature_importance is None:
            return None
            
        # Create a DataFrame with feature names and importance scores
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.feature_importance
        })
        
        # Sort by importance and get top N
        top_features = importance_df.sort_values('importance', ascending=False).head(top_n)
        
        return top_features
    
    def plot_feature_importance(self, top_n=10, save_path=None):
        """Plot feature importance"""
        top_features = self.get_feature_importance(top_n)
        if top_features is None:
            return
            
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=top_features)
        plt.title('Top Feature Importance for RTO Prediction')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def save_model(self, path=None):
        """Save the trained model and preprocessing objects"""
        if path:
            self.model_path = path
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
        # Save model and metadata
        model_data = {
            'model': self.model,
            'preprocessor': self.preprocessor,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'feature_importance': self.feature_importance,
            'training_history': self.training_history,
            'categorical_features': self.categorical_features,
            'numerical_features': self.numerical_features,
            'all_features': self.all_features
        }
        
        joblib.dump(model_data, self.model_path)
        print(f"Model saved to {self.model_path}")
    
    def load_model(self, path=None):
        """Load a trained model and preprocessing objects"""
        if path:
            self.model_path = path
            
        if os.path.exists(self.model_path):
            model_data = joblib.load(self.model_path)
            self.model = model_data['model']
            self.preprocessor = model_data['preprocessor']
            self.feature_names = model_data['feature_names']
            self.model_type = model_data['model_type']
            self.feature_importance = model_data['feature_importance']
            self.training_history = model_data.get('training_history', [])
            self.categorical_features = model_data.get('categorical_features', self.categorical_features)
            self.numerical_features = model_data.get('numerical_features', self.numerical_features)
            self.all_features = model_data.get('all_features', self.all_features)
            print(f"Model loaded from {self.model_path}")
            return True
        else:
            print(f"No model found at {self.model_path}")
            return False
    
    def cross_validate(self, training_data, target_column='rto_risk', cv=5):
        """Perform cross-validation on the model"""
        if not isinstance(training_data, pd.DataFrame):
            training_data = pd.DataFrame(training_data)
            
        X = training_data[self.all_features]
        y = training_data[target_column]
        
        # Create model pipeline
        self._create_model_pipeline()
        
        # Perform cross-validation
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='f1_weighted')
        
        print(f"Cross-validation scores: {scores}")
        print(f"Mean CV score: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        return scores.mean(), scores.std()

# Example usage:
if __name__ == "__main__":
    # Create sample training data with more examples
    sample_data = pd.DataFrame({
        'address_classification': ['Remote', 'Urban', 'Suburban', 'Remote', 'Urban', 'Suburban', 'Remote', 'Urban', 'Suburban', 'Remote', 'Urban', 'Suburban', 'Remote', 'Urban', 'Suburban'],
        'status': ['Pending', 'Shipped', 'Delivered', 'Pending', 'Shipped', 'Delivered', 'Pending', 'Shipped', 'Delivered', 'Pending', 'Shipped', 'Delivered', 'Pending', 'Shipped', 'Delivered'],
        'payment_method': ['COD', 'Online', 'COD', 'Online', 'COD', 'Online', 'COD', 'Online', 'COD', 'Online', 'COD', 'Online', 'COD', 'Online', 'COD'],
        'quantity': [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
        'city': ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad', 'Pune', 'Ahmedabad', 'Jaipur', 'Lucknow', 'Kanpur', 'Nagpur', 'Indore', 'Bhopal', 'Patna'],
        'state': ['Maharashtra', 'Delhi', 'Karnataka', 'Tamil Nadu', 'West Bengal', 'Telangana', 'Maharashtra', 'Gujarat', 'Rajasthan', 'UP', 'UP', 'Maharashtra', 'MP', 'MP', 'Bihar'],
        'pincode': ['400001', '110001', '560001', '600001', '700001', '500001', '411001', '380001', '302001', '226001', '208001', '440001', '452001', '462001', '800001'],
        'rto_risk': ['High', 'Medium', 'Low', 'High', 'Medium', 'Low', 'High', 'Medium', 'Low', 'High', 'Medium', 'Low', 'High', 'Medium', 'Low']
    })
    
    # Initialize and train the model
    predictor = RTOPredictor()
    predictor.train(sample_data, model_type='random_forest')
    
    # Save the model
    predictor.save_model()
    
    # Test prediction with multiple examples
    test_data = pd.DataFrame({
        'address_classification': ['Remote', 'Urban', 'Suburban'],
        'status': ['Pending', 'Shipped', 'Delivered'],
        'payment_method': ['COD', 'Online', 'COD'],
        'quantity': [1, 2, 3],
        'city': ['Mumbai', 'Delhi', 'Bangalore'],
        'state': ['Maharashtra', 'Delhi', 'Karnataka'],
        'pincode': ['400001', '110001', '560001']
    })
    
    predictions, probabilities = predictor.predict(test_data)
    print("\nTest Predictions:")
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        print(f"Order {i+1}:")
        print(f"Predicted RTO Risk: {pred}")
        print(f"Risk Probabilities: {prob}")
        print()
    
    # Plot feature importance
    predictor.plot_feature_importance() 