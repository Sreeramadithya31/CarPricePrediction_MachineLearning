"""CodeAlpha Task 3: Car Price Prediction.

Expected dataset: data/CarPrice_Assignment.csv (common automobile price dataset).
"""
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "CarPrice_Assignment.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
TARGET = "Selling_Price"

def main():
    if not DATA.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA}\n"
            "Place the CodeAlpha car-price CSV at data/CarPrice_Assignment.csv."
        )

    df = pd.read_csv(DATA)
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found. Available columns: {list(df.columns)}")

    # Drop identifier-like columns; they do not represent useful car specifications.
    drop_cols = [c for c in ["car_ID"] if c in df.columns]
    df = df.drop(columns=drop_cols)
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET])
    df = df[df[TARGET] > 0]

    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = X.select_dtypes(exclude=["number"]).columns.tolist()

    if not numeric and not categorical:
        raise ValueError("No usable feature columns were found.")

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric),
        ("categorical", categorical_pipeline, categorical)
    ])

    model = Pipeline([
        ("preprocess", preprocess),
        ("regressor", RandomForestRegressor(
            n_estimators=300, min_samples_leaf=1, random_state=42, n_jobs=-1
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)
    cv = cross_val_score(model, X, y, cv=5, scoring="r2", n_jobs=1)

    metrics = {
        "test_MAE": round(float(mae), 3),
        "test_RMSE": round(float(rmse), 3),
        "test_R2": round(float(r2), 4),
        "five_fold_CV_R2_mean": round(float(cv.mean()), 4),
        "five_fold_CV_R2_std": round(float(cv.std()), 4),
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "numeric_features": numeric,
        "categorical_features": categorical
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("\n--- Model evaluation ---")
    print(json.dumps(metrics, indent=2))

    results = pd.DataFrame({"actual_price": y_test, "predicted_price": predictions})
    results.to_csv(OUT / "predictions.csv", index=False)

    plt.figure(figsize=(7, 6))
    sns.scatterplot(x=y_test, y=predictions)
    low = min(float(y_test.min()), float(predictions.min()))
    high = max(float(y_test.max()), float(predictions.max()))
    plt.plot([low, high], [low, high], linestyle="--")
    plt.title("Actual vs Predicted Car Prices")
    plt.xlabel("Actual price")
    plt.ylabel("Predicted price")
    plt.tight_layout()
    plt.savefig(OUT / "actual_vs_predicted.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7, 5))
    residuals = y_test - predictions
    sns.histplot(residuals, kde=True)
    plt.title("Prediction Residual Distribution")
    plt.xlabel("Actual - predicted price")
    plt.tight_layout()
    plt.savefig(OUT / "residual_distribution.png", dpi=300)
    plt.close()

    print("\nOutputs saved in:", OUT)
    print("Task 3 completed successfully.")

if __name__ == "__main__":
    main()
