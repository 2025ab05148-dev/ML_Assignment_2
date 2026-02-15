import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# ==============================
# 1️⃣ Load Dataset
# ==============================
df = pd.read_csv("healthcare-dataset-stroke-data.csv")

# ==============================
# 2️⃣ Preprocessing
# ==============================

# Drop id
if "id" in df.columns:
    df = df.drop("id", axis=1)

# Fill missing bmi
df["bmi"] = df["bmi"].fillna(df["bmi"].mean())

# Encode categorical columns
for col in df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# ==============================
# 3️⃣ Features & Target
# ==============================
X = df.drop("stroke", axis=1)
y = df["stroke"]

# ==============================
# 4️⃣ Create models folder
# ==============================
os.makedirs("models", exist_ok=True)

# 🔥 SAVE FEATURE COLUMN ORDER (VERY IMPORTANT)
joblib.dump(X.columns.tolist(), "models/feature_columns.pkl")

# ==============================
# 5️⃣ Train Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# 6️⃣ Scaling
# ==============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Save scaler
joblib.dump(scaler, "models/scaler.pkl")

# ==============================
# 7️⃣ Define Models
# ==============================
models = {
    "logistic": LogisticRegression(max_iter=1000),
    "decision_tree": DecisionTreeClassifier(),
    "knn": KNeighborsClassifier(),
    "naive_bayes": GaussianNB(),
    "random_forest": RandomForestClassifier(),
    "xgboost": XGBClassifier(eval_metric="logloss")
}

# ==============================
# 8️⃣ Train & Save Models
# ==============================
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    joblib.dump(model, f"models/{name}.pkl")

print("✅ Stroke models trained and saved successfully")
print("✅ feature_columns.pkl saved")
