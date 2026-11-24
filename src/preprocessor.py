import pandas as pd
from sklearn.preprocessing import MinMaxScaler

FEATURE_COLS = [
    "Poverty_Rate", "Food_Poverty", "Under5_Pct", "Elderly_Pct",
    "Disability_Idx", "Flood_Inundation_m", "Drought_Severity", "Dist_Health_km"
]

DEFAULT_AHP_WEIGHTS = {
    "Poverty_Rate": 0.22,
    "Food_Poverty": 0.13,
    "Under5_Pct": 0.10,
    "Elderly_Pct": 0.08,
    "Disability_Idx": 0.12,
    "Flood_Inundation_m": 0.15,
    "Drought_Severity": 0.12,
    "Dist_Health_km": 0.08,
}


class VulnerabilityPreprocessor:
    """Normalizes indicators and computes AHP multi-criteria vulnerability scores."""

    def __init__(self, weights: dict = None):
        self.scaler = MinMaxScaler()
        self.weights = weights if weights is not None else DEFAULT_AHP_WEIGHTS

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        norm_vals = self.scaler.fit_transform(out[FEATURE_COLS])
        norm_df = pd.DataFrame(norm_vals, columns=FEATURE_COLS, index=out.index)

        total_w = sum(self.weights.values())
        norm_weights = {k: v / total_w for k, v in self.weights.items()}

        out["Vulnerability_Score"] = sum(
            norm_df[col] * norm_weights[col] for col in FEATURE_COLS
        ).round(4)

        out["Priority_Tier"] = pd.qcut(
            out["Vulnerability_Score"], q=3, labels=["Moderate", "High", "Critical"]
        )
        return out