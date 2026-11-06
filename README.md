# Predicción de churn — Telco Customer Churn

Pipeline de scikit-learn (feature engineering → preprocesamiento → HistGradientBoosting optimizado con GridSearchCV) para predecir abandono de clientes.

- `churn_notebook.ipynb`: EDA, CV, GridSearch, evaluación y exportación
- `data/telco_churn.csv`: dataset
- `churn_pipeline.pkl`: pipeline entrenado (requiere `churn_features.py` y `churn_model.py` en el path)

Resultado en test: ROC-AUC 0.845, F1 (churn) 0.633, Recall 0.765.

```python
import joblib
model = joblib.load("churn_pipeline.pkl")
model.predict_proba(df_clientes)
```
