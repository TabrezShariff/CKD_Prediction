import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import joblib
import os
import warnings
warnings.filterwarnings('ignore')
from collections import Counter

print("Starting CKD Model Training with Data Quality Improvements...")
print("="*60)

# Load the dataset
data = pd.read_csv('data\Kidney_data.csv')
print(f"Initial dataset shape: {data.shape}")

# Data Quality Assessment
print("\n1. DATA QUALITY ASSESSMENT:")
print("-" * 30)
print(f"Total missing values: {data.isnull().sum().sum()}")
print(f"Duplicate rows: {data.duplicated().sum()}")

# Remove duplicate rows
data = data.drop_duplicates()
print(f"Shape after removing duplicates: {data.shape}")

# Handle missing values represented as '?' or other strings
print("\n2. HANDLING MISSING VALUES:")
print("-" * 30)
for column in data.columns:
    if data[column].dtype == 'object':
        # Replace various representations of missing values
        data[column] = data[column].replace(['?', '', ' ', 'nan', 'NaN', 'NULL'], np.nan)
        missing_count = data[column].isnull().sum()
        if missing_count > 0:
            print(f"{column}: {missing_count} missing values")

# Convert numeric columns that might be stored as strings
numeric_cols_to_convert = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
for col in numeric_cols_to_convert:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

# Separate numeric and categorical columns
numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
categorical_columns = data.select_dtypes(exclude=[np.number]).columns.tolist()

# Remove 'id' and 'classification' from feature columns
if 'id' in numeric_columns:
    numeric_columns.remove('id')
if 'classification' in categorical_columns:
    categorical_columns.remove('classification')

print(f"Numeric columns: {numeric_columns}")
print(f"Categorical columns: {categorical_columns}")

# Advanced imputation strategy
print("\n3. IMPUTING MISSING VALUES:")
print("-" * 30)

# For numeric columns - use median for better handling of outliers
numeric_imputer = SimpleImputer(strategy='median')
if numeric_columns:
    data[numeric_columns] = numeric_imputer.fit_transform(data[numeric_columns])
    print(f"Imputed {len(numeric_columns)} numeric columns with median values")

# For categorical columns - use most frequent
categorical_imputer = SimpleImputer(strategy='most_frequent')
if categorical_columns:
    data[categorical_columns] = categorical_imputer.fit_transform(data[categorical_columns])
    print(f"Imputed {len(categorical_columns)} categorical columns with most frequent values")

# Encode categorical variables
print("\n4. ENCODING CATEGORICAL VARIABLES:")
print("-" * 30)
label_encoders = {}
for column in categorical_columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column].astype(str))
    label_encoders[column] = le
    print(f"Encoded {column}: {len(le.classes_)} unique values")

# Remove outliers using IQR method for key medical parameters
print("\n5. OUTLIER DETECTION AND HANDLING:")
print("-" * 30)
key_medical_params = ['age', 'bp', 'bgr', 'bu', 'sc', 'hemo']
outliers_removed = 0

for col in key_medical_params:
    if col in data.columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers_before = len(data)
        data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
        outliers_after = len(data)
        removed = outliers_before - outliers_after
        outliers_removed += removed
        
        if removed > 0:
            print(f"{col}: Removed {removed} outliers")

print(f"Total outliers removed: {outliers_removed}")
print(f"Final dataset shape: {data.shape}")

# Prepare features and target
X = data.drop(['id', 'classification'], axis=1, errors='ignore')
y = data['classification']

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target distribution:\n{y.value_counts()}")

# Define clinically important features for CKD prediction
clinically_important_features = [
    'sc',    # Serum Creatinine - most important
    'bu',    # Blood Urea - direct kidney function
    'bgr',   # Blood Glucose Random - diabetes indicator
    'bp',    # Blood Pressure - hypertension
    'hemo',  # Hemoglobin - anemia in CKD
    'al',    # Albumin - protein in urine
    'age',   # Age - demographic factor
    'htn',   # Hypertension - major risk factor
    'dm',    # Diabetes Mellitus - leading cause
    'pcv'    # Packed Cell Volume - anemia indicator
]

# Ensure we only use features that exist in the dataset
available_features = [feat for feat in clinically_important_features if feat in X.columns]
missing_features = [feat for feat in clinically_important_features if feat not in X.columns]

print(f"\n6. FEATURE SELECTION:")
print("-" * 30)
print(f"Clinically important features available: {len(available_features)}")
print(f"Available features: {available_features}")
if missing_features:
    print(f"Missing features: {missing_features}")

# Use hybrid approach: clinical importance + statistical selection
if len(available_features) >= 8:  # Use clinical features if we have enough
    selected_features = available_features[:10]  # Take top 10
    X_selected = X[selected_features]
    print("Using clinically important features")
else:
    # Fall back to statistical feature selection
    selector = SelectKBest(mutual_info_classif, k=min(10, X.shape[1]))
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()].tolist()
    X_selected = X[selected_features]
    print("Using statistical feature selection")

print(f"Final selected features: {selected_features}")

class_counts = Counter(y)

# Check if all classes have at least 2 samples
if all(count >= 2 for count in class_counts.values()):
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42, stratify=y
    )
    print("Used stratified split.")
else:
    print("Warning: Some classes have less than 2 samples. Falling back to random split.")
    print(f"Class distribution: {class_counts}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42
    )
    print("Used random split (no stratify).")

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter tuning for Random Forest
print("\n7. HYPERPARAMETER TUNING:")
print("-" * 30)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 15, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'class_weight': ['balanced', None]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

best_model = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Train the final model
model = best_model
model.fit(X_train_scaled, y_train)

# Evaluate the model
print("\n8. MODEL EVALUATION:")
print("-" * 30)

y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"Test Accuracy: {accuracy:.4f}")
print(f"ROC AUC Score: {roc_auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Cross-validation with the best model
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f"\nCross-validation scores: {cv_scores}")
print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Feature importance analysis
print("\n9. FEATURE IMPORTANCE ANALYSIS:")
print("-" * 30)
importances = model.feature_importances_
feature_importances = pd.DataFrame({
    'feature': selected_features, 
    'importance': importances
}).sort_values('importance', ascending=False)

print("Feature Importances:")
for idx, row in feature_importances.iterrows():
    print(f"{row['feature']:12}: {row['importance']:.4f}")

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Save the model, scaler, and selected features
joblib.dump(model, 'models/kidney_disease_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(selected_features, 'models/selected_features.pkl')

# Save additional metadata
metadata = {
    'selected_features': selected_features,
    'feature_importances': feature_importances.to_dict('records'),
    'model_performance': {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    },
    'best_params': grid_search.best_params_,
    'training_samples': len(X_train),
    'test_samples': len(X_test)
}

joblib.dump(metadata, 'models/model_metadata.pkl')

print("\n" + "="*60)
print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("="*60)
print(f"✓ Model saved to: models/kidney_disease_model.pkl")
print(f"✓ Scaler saved to: models/scaler.pkl")
print(f"✓ Selected features saved to: models/selected_features.pkl")
print(f"✓ Metadata saved to: models/model_metadata.pkl")
print(f"\nFinal Model Performance:")
print(f"  - Accuracy: {accuracy:.4f}")
print(f"  - ROC AUC: {roc_auc:.4f}")
print(f"  - CV Score: {cv_scores.mean():.4f}")
print(f"  - Features: {len(selected_features)}")
print("\nSelected Features for App:")
for i, feature in enumerate(selected_features, 1):
    print(f"  {i}. {feature}")