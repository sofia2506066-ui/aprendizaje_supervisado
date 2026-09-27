from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "titanic_clean.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "titanic_model.joblib"


def build_preprocessor(numeric_features, categorical_features):
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )


def main():
    df = pd.read_csv(DATA_PATH)
    target = "Survived"
    features = df.drop(columns=[target])
    labels = df[target]

    numeric_features = features.select_dtypes(include="number").columns.tolist()
    categorical_features = features.select_dtypes(exclude="number").columns.tolist()
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = {}
    best_score = -1
    trained_models = {}

    for name, estimator in candidates.items():
        pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("classifier", estimator)]
        )
        pipeline.fit(x_train, y_train)
        trained_models[name] = pipeline
        predictions = pipeline.predict(x_test)
        score = f1_score(y_test, predictions)
        results[name] = {
            "accuracy": accuracy_score(y_test, predictions),
            "f1": score,
        }
        print(f"\n{name}")
        print(classification_report(y_test, predictions, zero_division=0))
        if score > best_score:
            best_name = name
            best_score = score

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(trained_models, MODEL_PATH)
    print(f"Mejor modelo: {best_name}")
    print(f"Modelo guardado en: {MODEL_PATH}")
    print(f"Comparacion: {results}")


if __name__ == "__main__":
    main()