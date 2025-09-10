##############################################
# E-commerce Return Rate Reduction Analysis
# Decision Tree Modeling
##############################################

# ========== 1) Imports ==========
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, RocCurveDisplay
)
from sklearn.utils import class_weight
import joblib
import warnings
warnings.filterwarnings("ignore")

# ========== 2) Load Data ==========
DATA_PATH = "C:\\Users\\HP\\Downloads\\ecommerce_returns.csv.csv"   # <--- change if needed
df = pd.read_csv("C:\\Users\\HP\\Downloads\\ecommerce_returns.csv.csv")
print("Shape:", df.shape)
print(df.head(3))

# ========== 3) Light Schema Check / Renames ==========
# Adjust to your dataset. Typical columns in a returns dataset might include:
# 'Order_ID','Customer_ID','Order_Date','Ship_Date','Category','Sub_Category',
# 'Product_ID','Price','Quantity','Discount','Shipping_Cost','Region','Country',
# 'Marketing_Channel','Return_Status','Return_Reason','Return_Date'
expected_cols = ['Return_Status']  # minimally required target
missing = [c for c in expected_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Create binary target: 1 if Returned else 0
df['Return_Flag'] = (df['Return_Status'].astype(str).str.strip().str.lower() == 'returned').astype(int)

# Optional: parse dates if present
for col in ['Order_Date', 'Ship_Date', 'Return_Date']:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

# Optional engineered features
if {'Order_Date','Ship_Date'}.issubset(df.columns):
    df['Ship_Days'] = (df['Ship_Date'] - df['Order_Date']).dt.days
if {'Return_Date','Order_Date'}.issubset(df.columns):
    df['Return_Processing_Days'] = (df['Return_Date'] - df['Order_Date']).dt.days

# Strip strings in categorical-like columns
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].astype(str).str.strip()

# Drop obvious ID-like columns (they usually don’t help prediction)
drop_like = [c for c in df.columns if c.lower() in {'order_id','product_id','customer_id'}]
df = df.drop(columns=drop_like, errors='ignore')

# ========== 4) Quick EDA Visuals ==========
# Identify numeric and categorical columns BEFORE encoding
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# remove target from numeric list
numeric_cols = [c for c in numeric_cols if c != 'Return_Flag']

categorical_cols = df.select_dtypes(include=['object','category']).columns.tolist()
categorical_cols = [c for c in categorical_cols if c not in ['Return_Status']]  # we already transformed it

print("\nNumeric columns:", numeric_cols[:10], "..." if len(numeric_cols)>10 else "")
print("Categorical columns:", categorical_cols[:10], "..." if len(categorical_cols)>10 else "")

# 4a) Histograms of numeric features
if numeric_cols:
    df[numeric_cols].hist(bins=30, figsize=(14, 10))
    plt.suptitle("Numeric Feature Distributions", y=1.02)
    plt.tight_layout()
    plt.show()

# 4b) Scatter plot: choose two common numeric features if they exist
x_feat = 'Price' if 'Price' in df.columns else (numeric_cols[0] if numeric_cols else None)
y_feat = 'Quantity' if 'Quantity' in df.columns else (numeric_cols[1] if len(numeric_cols) > 1 else None)

if x_feat and y_feat:
    plt.figure(figsize=(7,5))
    sns.scatterplot(data=df, x=x_feat, y=y_feat, hue='Return_Flag', alpha=0.6)
    plt.title(f"{x_feat} vs {y_feat} by Return_Flag")
    plt.tight_layout()
    plt.show()

# 4c) Correlation heatmap for numeric features
if len(numeric_cols) >= 2:
    plt.figure(figsize=(10,8))
    corr = df[numeric_cols + ['Return_Flag']].corr(numeric_only=True)
    sns.heatmap(corr, annot=False, cmap='coolwarm', center=0)
    plt.title("Correlation Heatmap (Numeric)")
    plt.tight_layout()
    plt.show()

# ========== 5) Train/Test Split ==========
X = df.drop(columns=['Return_Flag', 'Return_Status'])
y = df['Return_Flag']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTrain size:", X_train.shape, " Test size:", X_test.shape)
print("Positives in train:", y_train.sum(), "of", len(y_train))

# ========== 6) Preprocessing & Pipeline ==========
numeric_features = [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])]
categorical_features = [c for c in X_train.columns if c not in numeric_features]

numeric_transformer = SimpleImputer(strategy='median')
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy='most_frequent')),
    ("onehot", OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ],
    remainder='drop'
)

clf = DecisionTreeClassifier(random_state=42, class_weight='balanced')  # helps if class imbalance

pipe = Pipeline(steps=[("prep", preprocess),
                      ("model", clf)])

# ========== 7) Hyperparameter Tuning (quick) ==========
param_grid = {
    "model__max_depth": [4, 6, 8, 10, None],
    "model__min_samples_split": [2, 10, 20],
    "model__min_samples_leaf": [1, 5, 10],
    "model__criterion": ["gini", "entropy"]
}

grid = GridSearchCV(
    pipe,
    param_grid=param_grid,
    scoring="f1",         # change to 'roc_auc' if you prefer AUC
    cv=5,
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train, y_train)
print("\nBest params:", grid.best_params_)
best_model = grid.best_estimator_

# ========== 8) Evaluation ==========
def evaluate(model, X_tr, y_tr, X_te, y_te, label="Model"):
    ytr_pred = model.predict(X_tr)
    yte_pred = model.predict(X_te)

    # Some metrics also need probabilities
    if hasattr(model, "predict_proba"):
        ytr_proba = model.predict_proba(X_tr)[:,1]
        yte_proba = model.predict_proba(X_te)[:,1]
    else:
        ytr_proba = None
        yte_proba = None

    print(f"\n===== {label} - Train =====")
    print(classification_report(y_tr, ytr_pred, digits=4))
    print("Accuracy:", accuracy_score(y_tr, ytr_pred))
    if ytr_proba is not None:
        print("ROC AUC:", roc_auc_score(y_tr, ytr_proba))

    print(f"\n===== {label} - Test =====")
    print(classification_report(y_te, yte_pred, digits=4))
    print("Accuracy:", accuracy_score(y_te, yte_pred))
    if yte_proba is not None:
        print("ROC AUC:", roc_auc_score(y_te, yte_proba))

    # Confusion matrix
    cm = confusion_matrix(y_te, yte_pred)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{label} - Confusion Matrix (Test)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

    # ROC curve
    if yte_proba is not None:
        RocCurveDisplay.from_predictions(y_te, yte_proba)
        plt.title(f'{label} - ROC Curve (Test)')
        plt.tight_layout()
        plt.show()

evaluate(best_model, X_train, y_train, X_test, y_test, label="Decision Tree (Tuned)")

# ========== 9) Feature Importance ==========
# Get feature names after preprocessing
ohe = best_model.named_steps['prep'].named_transformers_['cat'].named_steps['onehot']
cat_feature_names = ohe.get_feature_names_out(categorical_features) if len(categorical_features) else np.array([])
final_feature_names = np.r_[numeric_features, cat_feature_names]

importances = best_model.named_steps['model'].feature_importances_
fi = pd.DataFrame({'feature': final_feature_names, 'importance': importances})
fi = fi.sort_values('importance', ascending=False).head(20)

plt.figure(figsize=(8,6))
sns.barplot(data=fi, x='importance', y='feature')
plt.title("Top 20 Feature Importances (Decision Tree)")
plt.tight_layout()
plt.show()

# ========== 10) Optional: Visualize the Tree (shallow depth only) ==========
# Warning: Large trees are hard to render. Use a small depth for illustration.
small_tree = DecisionTreeClassifier(
    random_state=42, max_depth=3, class_weight='balanced',
    criterion=grid.best_params_['model__criterion']
)
small_pipe = Pipeline([("prep", preprocess), ("model", small_tree)])
small_pipe.fit(X_train, y_train)

plt.figure(figsize=(16,8))
# plot_tree needs feature names post-transform:
plot_tree(
    small_pipe.named_steps['model'],
    feature_names=final_feature_names,
    class_names=['Not Returned', 'Returned'],
    filled=True, impurity=False, rounded=True
)
plt.title("Interpretable Small Decision Tree (Depth=3)")
plt.show()

# ========== 11) Save the Model ==========
joblib.dump(best_model, "decision_tree_returns_pipeline.joblib")
print("\nSaved model to decision_tree_returns_pipeline.joblib")