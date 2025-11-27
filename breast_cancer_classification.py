import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier

def load_dataset():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    df = pd.concat([X, y], axis=1)
    return df, X, y, data

def explore_data(df):
    print(df.head())
    print(df.describe())
    print(df["target"].value_counts())

    plt.figure(figsize=(6, 4))
    sns.countplot(x="target", data=df)
    plt.title("Class Distribution")
    plt.tight_layout()
    plt.show()

    corr = df.corr(numeric_only=True)
    top_features = corr["target"].abs().sort_values(ascending=False).head(12).index

    plt.figure(figsize=(10, 8))
    sns.heatmap(df[top_features].corr(), cmap="coolwarm", annot=False)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.title("Top Feature Correlations")
    plt.tight_layout()
    plt.show()

def split_dataset(X, y):
    return train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

def train_simple_model(X_train, y_train):
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=150,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        ))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

def evaluate_model(model, X_test, y_test, target_names):
    preds = model.predict(X_test)
    print("F1 Score:", f1_score(y_test, preds))
    print(classification_report(y_test, preds, target_names=target_names))

    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=target_names)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df, X, y, data_info = load_dataset()
    explore_data(df)
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    model = train_simple_model(X_train, y_train)
    evaluate_model(model, X_test, y_test, target_names=data_info.target_names)
