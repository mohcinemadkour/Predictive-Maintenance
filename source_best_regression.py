import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestRegressor

# Define paths
DATA_TRAIN_PATH = 'data/train.csv'
MODEL_DIR = 'Models-regression'
MODEL_FILE = os.path.join(MODEL_DIR, 'best_rf_regressor.pkl')

def source_best_model():
    # 1. Create directory if it doesn't exist
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        print(f"Created directory: {MODEL_DIR}")

    # 2. Load training data
    if not os.path.exists(DATA_TRAIN_PATH):
        print(f"Error: {DATA_TRAIN_PATH} not found.")
        return

    df_train = pd.read_csv(DATA_TRAIN_PATH)
    
    # 3. Define features (original features as used in the best RF model in notebook)
    features_orig = ['setting1','setting2','setting3','s1','s2','s3','s4','s5','s6','s7','s8','s9','s10','s11','s12','s13','s14','s15','s16','s17','s18','s19','s20','s21']
    X_train = df_train[features_orig]
    y_train = df_train['ttf']

    # 4. Initialize and train the best model (Random Forest)
    # Configuration from notebook: n_estimators=100, max_features=3, max_depth=4, n_jobs=-1, random_state=1
    print("Training Random Forest Regressor...")
    rf = RandomForestRegressor(n_estimators=100, max_features=3, max_depth=4, n_jobs=-1, random_state=1)
    rf.fit(X_train, y_train)

    # 5. Save the model
    joblib.dump(rf, MODEL_FILE)
    print(f"Model saved to: {MODEL_FILE}")

if __name__ == "__main__":
    source_best_model()
