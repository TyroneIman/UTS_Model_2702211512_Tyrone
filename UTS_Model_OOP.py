import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro
import warnings
import pickle
import os

from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

class BookingModel:
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath)
        self.label_encoders = {}
        self.scaler = RobustScaler()
        self.model = None

    def inspect_data(self):
        print("First 5 rows:\n", self.df.head())
        print("\nShape:", self.df.shape)
        print("\nColumns:", self.df.columns)
        print("\nInfo:")
        print(self.df.info())

    def clean_data(self):
        self.df.dropna(inplace=True)
        self.df.drop_duplicates(inplace=True)

    def encode_features(self):
        categorical_cols = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type']
        binary_map = {'No': 0, 'Yes': 1}

        for col in categorical_cols:
            if col in self.df.columns:
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col])
                self.label_encoders[col] = le

        if 'repeated_guest' in self.df.columns:
            self.df['repeated_guest'] = self.df['repeated_guest'].map(binary_map)

        if 'booking_status' in self.df.columns:
            le = LabelEncoder()
            self.df['booking_status'] = le.fit_transform(self.df['booking_status'])
            self.label_encoders['booking_status'] = le

    def test_normality(self):
        print("\nShapiro-Wilk Normality Test:")
        numerical_cols = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        for col in numerical_cols:
            stat, p = shapiro(self.df[col])
            print(f"{col}: W={stat:.4f}, p-value={p:.4f}")

    def visualize(self):
        plt.figure(figsize=(8, 6))
        for status in self.df['booking_status'].unique():
            subset = self.df[self.df['booking_status'] == status]
            plt.scatter(subset['no_of_adults'], subset['no_of_weekend_nights'], label=status, alpha=0.6)
        plt.xlabel('Number of Adults')
        plt.ylabel('Number of Weekend Nights')
        plt.title('Booking Status Distribution')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def prepare_data(self):
        X = self.df.drop(columns=['booking_status', 'Booking_ID'], errors='ignore')
        y = self.df['booking_status']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

    def train_models(self):
        print("\nTraining Random Forest and XGBoost...")
        rf = RandomForestClassifier(max_depth=4, random_state=42)
        xgb = XGBClassifier(max_depth=4, use_label_encoder=False, eval_metric='mlogloss', random_state=42)

        rf.fit(self.X_train_scaled, self.y_train)
        self.y_train = self.y_train.astype(int)  # Ensure labels are int for XGBoost
        xgb.fit(self.X_train_scaled, self.y_train)

        print("\nRandom Forest Report:")
        print(classification_report(self.y_test, rf.predict(self.X_test_scaled)))

        print("\nXGBoost Report:")
        print(classification_report(self.y_test, xgb.predict(self.X_test_scaled)))

        self.model = xgb

    def tune_model(self):
        print("\nTuning XGBoost with GridSearchCV...")
        param_grid = {
            'max_depth': [2, 4, 6],
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1]
        }
        grid_search = GridSearchCV(XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
                                   param_grid, scoring='f1_macro', cv=5)
        grid_search.fit(self.X_train_scaled, self.y_train)
        self.model = grid_search.best_estimator_

        print("\nBest Parameters:", grid_search.best_params_)
        print("\nClassification Report:")
        print(classification_report(self.y_test, self.model.predict(self.X_test_scaled)))

    def save_model(self, filename="best_xgboost_model.pkl"):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {filename}")


if __name__ == '__main__':
    bm = BookingModel('Dataset_B_hotel.csv')
    bm.inspect_data()
    bm.clean_data()
    bm.encode_features()
    bm.test_normality()
    bm.visualize()
    bm.prepare_data()
    bm.train_models()
    bm.tune_model()
    bm.save_model()
