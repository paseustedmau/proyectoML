import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

SERVICE_COLS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies"]


class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """Limpia TotalCharges y crea variables derivadas para el dataset Telco Churn."""

    def fit(self, X, y=None):
        self.feature_names_in_ = np.array(X.columns)
        return self

    def transform(self, X):
        X = X.copy()
        # TotalCharges viene como texto con espacios en blanco para clientes nuevos
        X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")
        X.loc[X["tenure"] == 0, "TotalCharges"] = 0.0

        X["n_services"] = sum((X[c] == "Yes").astype(int) for c in SERVICE_COLS)
        X["has_internet"] = (X["InternetService"] != "No").astype(int)
        X["has_phone_only"] = ((X["InternetService"] == "No") & (X["PhoneService"] == "Yes")).astype(int)
        X["avg_monthly_charge"] = np.where(X["tenure"] > 0, X["TotalCharges"] / X["tenure"].replace(0, 1), X["MonthlyCharges"])
        X["charge_diff"] = X["MonthlyCharges"] - X["avg_monthly_charge"]
        X["tenure_group"] = pd.cut(X["tenure"], bins=[-1, 12, 24, 48, np.inf],
                                   labels=["0-12", "13-24", "25-48", "49+"]).astype(str)
        X["is_autopay"] = X["PaymentMethod"].str.contains("automatic", case=False, na=False).astype(int)
        X["no_protection"] = ((X["InternetService"] != "No") &
                              (X["OnlineSecurity"] != "Yes") & (X["TechSupport"] != "Yes")).astype(int)
        X["monthly_contract"] = (X["Contract"] == "Month-to-month").astype(int)
        return X
