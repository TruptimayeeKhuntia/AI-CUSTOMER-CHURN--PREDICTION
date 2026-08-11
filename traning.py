from dbm import dumb

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split , KFold,cross_val_score , StratifiedKFold , StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from joblib import dump
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score,roc_auc_score, classification_report 


pd.set_option("display.max_columns", None)

df = pd.read_csv(r"C:\Users\TRUPTIMAYEE\AppData\Local\Packages\5319275A.WhatsAppDesktop_cv1g1gvanyjgm\LocalState\sessions\3BB514405FE9D531C1E0FF45DC5F6E4855D9E8DF\transfers\2026-32\housing.csv")
print(df.head())
print("Shape:", df.shape)
print("Columns:", list(df.columns))

print("\nMissing values:")
print(df.isnull().sum())

print("\nTarget summary:")
print(df["median_house_value"].describe())

df = df.dropna(subset=["total_bedrooms"])

X = df.drop(columns=["median_house_value", "ocean_proximity"])
y = df["median_house_value"]

print("\nX shape:", X.shape)
print("y shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nX_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)


pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


dump(pipeline, "model_dir/housing_model.joblib")