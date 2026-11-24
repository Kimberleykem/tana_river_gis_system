import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from src.preprocessor import FEATURE_COLS


class PrioritizationMLModel:
    """Random Forest Classifier for automated disaster relief priority prediction."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = RandomForestClassifier(
            n_estimators=200, max_depth=5, min_samples_leaf=4, random_state=random_state
        )
        self.label_encoder = LabelEncoder()
        self.metrics = {}

    def train_and_evaluate(self, df_wards: pd.DataFrame, preprocessor) -> dict:
        np.random.seed(self.random_state)
        records = []
        for _, row in df_wards.iterrows():
            for _ in range(40):
                records.append({
                    "Poverty_Rate": np.clip(np.random.normal(row["Poverty_Rate"], 2.5), 40, 90),
                    "Food_Poverty": np.clip(np.random.normal(row["Food_Poverty"], 2.5), 25, 80),
                    "Under5_Pct": np.clip(np.random.normal(row["Under5_Pct"], 0.8), 10, 25),
                    "Elderly_Pct": np.clip(np.random.normal(row["Elderly_Pct"], 0.4), 1.5, 7.0),
                    "Disability_Idx": np.clip(np.random.normal(row["Disability_Idx"], 0.05), 0.3, 1.0),
                    "Flood_Inundation_m": np.random.choice(
                        [0.5, 1.0, 5.0],
                        p=[0.7, 0.2, 0.1] if row["Flood_Inundation_m"] == 0.5
                        else ([0.2, 0.6, 0.2] if row["Flood_Inundation_m"] == 1.0 else [0.1, 0.2, 0.7])
                    ),
                    "Drought_Severity": np.clip(np.random.normal(row["Drought_Severity"], 0.05), 0.3, 1.0),
                    "Dist_Health_km": np.clip(np.random.normal(row["Dist_Health_km"], 2.0), 1.0, 35.0),
                })

        df_ml = pd.DataFrame(records)
        norm_vals = preprocessor.scaler.transform(df_ml[FEATURE_COLS])
        norm_df = pd.DataFrame(norm_vals, columns=FEATURE_COLS)
        total_w = sum(preprocessor.weights.values())
        df_ml["Score"] = sum(norm_df[c] * (preprocessor.weights[c] / total_w) for c in FEATURE_COLS)
        df_ml["Target_Tier"] = pd.cut(
            df_ml["Score"], bins=[-np.inf, 0.48, 0.62, np.inf], labels=["Moderate", "High", "Critical"]
        )

        X = df_ml[FEATURE_COLS]
        y = self.label_encoder.fit_transform(df_ml["Target_Tier"])
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=self.random_state, stratify=y
        )

        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)

        self.metrics = {
            "train_accuracy": round(self.model.score(X_train, y_train) * 100, 2),
            "test_accuracy": round(accuracy_score(y_test, preds) * 100, 2),
            "macro_f1": round(f1_score(y_test, preds, average="macro") * 100, 2),
            "feature_importances": dict(zip(FEATURE_COLS, self.model.feature_importances_.round(4))),
        }
        return self.metrics

    def predict_wards(self, df_wards: pd.DataFrame) -> np.ndarray:
        preds = self.model.predict(df_wards[FEATURE_COLS])
        return self.label_encoder.inverse_transform(preds)