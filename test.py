import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
import glob



# Load and concatenate training data
train_files = glob.glob("data/sharechat_recsys2023_data/train/*.csv")  # Match all CSV files in the 'train' folder
train_data = pd.concat([pd.read_csv(file, sep='\t') for file in train_files])

# Split into features and labels
X = train_data.iloc[:, 1:-2]  # Exclude the first column (RowId) and the last two columns (labels)
y_click = train_data["is_clicked"]
y_install = train_data["is_installed"]

# Train-test split
X_train, X_test, y_click_train, y_click_test, y_install_train, y_install_test = train_test_split(
    X, y_click, y_install, test_size=0.2, random_state=42)

# Preprocessing
num_features = list(range(42, 80))  # Numerical features
cat_features = list(range(2, 33))  # Categorical features

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)])

# Model
model_click = RandomForestClassifier(random_state=42)
model_install = RandomForestClassifier(random_state=42)

# Pipeline
pipeline_click = Pipeline(steps=[('preprocessor', preprocessor), ('model', model_click)])
pipeline_install = Pipeline(steps=[('preprocessor', preprocessor), ('model', model_install)])

# Train the models
pipeline_click.fit(X_train, y_click_train)
pipeline_install.fit(X_train, y_install_train)

# Evaluate the models
y_click_pred = pipeline_click.predict(X_test)
y_install_pred = pipeline_install.predict(X_test)

print("Click Model Evaluation:")
print(classification_report(y_click_test, y_click_pred))

print("Install Model Evaluation:")
print(classification_report(y_install_test, y_install_pred))

# Custom evaluation metric (F1-score)
f1_click = f1_score(y_click_test, y_click_pred)
f1_install = f1_score(y_install_test, y_install_pred)

print("Click Model F1-score:", f1_click)
print("Install Model F1-score:", f1_install)
