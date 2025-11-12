import pandas as pd
import os
import config
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv(config.PROCESSED_DATA_FILE)

def severity_score(score):
    if score == 0:
        return 0
    elif score <= 2:
        return 1
    else:
        return 2

df['SEVERITY_CLASS'] = df['SEVERITY_SCORE'].apply(severity_score)

numeric_features = ['LATITUDE', 'LONGITUDE', 'YEAR', 'MONTH', 'DAY_OF_WEEK', 'HOUR']
categorical_features = ['BOROUGH', 'VEHICLE_TYPE_CODE_1', 'VEHICLE_TYPE_CODE_2',
                        'CONTRIBUTING_FACTOR_VEHICLE_1', 'CONTRIBUTING_FACTOR_VEHICLE_2']

X = df[numeric_features + categorical_features]
y = df['SEVERITY_CLASS']

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
)

numeric_transformer = StandardScaler()
categorical_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

X_train_p = preprocessor.fit_transform(X_train)
X_val_p = preprocessor.transform(X_val)

xgb_model = XGBClassifier(
    objective='multi:softprob',
    eval_metric='mlogloss',
    num_class=3,
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method='hist',
    random_state=config.RANDOM_STATE
)

xgb_model.fit(X_train_p, y_train)

y_val_pred = xgb_model.predict(X_val_p)

acc = accuracy_score(y_val, y_val_pred)
f1 = f1_score(y_val, y_val_pred, average='weighted')
report = classification_report(y_val, y_val_pred, zero_division=0)

print("Accuracy:", acc)
print("F1-Score:", f1)
print("Classification Report:\n", report)

cm = confusion_matrix(y_val, y_val_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('XGBoost Confusion Matrix')
os.makedirs(config.FIGURES_DIR, exist_ok=True)
plt.savefig(config.FIGURES_DIR / 'xgb_confusion_matrix.png')
plt.close()
