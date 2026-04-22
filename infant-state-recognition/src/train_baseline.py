import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.feature_engineering import load_data
from src.evaluate import evaluate_model

def main():
    print("Loading Data for Baseline Model (1D features)...")
    X, y, class_mapping = load_data(data_dir="data/raw", augment=False, feature_type='ml')
    
    if len(X) == 0:
        print("No data found. Please run generate_dummy_data.py first.")
        return
        
    class_names = [k for k, v in sorted(class_mapping.items(), key=lambda item: item[1])]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Training Logistic Regression Baseline...")
    model_lr = LogisticRegression(max_iter=1000)
    model_lr.fit(X_train_scaled, y_train)
    
    print("Evaluating Logistic Regression...")
    y_pred_lr = model_lr.predict(X_test_scaled)
    evaluate_model(y_test, y_pred_lr, "Baseline_LogReg", class_names)
    
    print("Training Decision Tree Baseline...")
    model_dt = DecisionTreeClassifier(random_state=42)
    model_dt.fit(X_train_scaled, y_train)
    
    print("Evaluating Decision Tree...")
    y_pred_dt = model_dt.predict(X_test_scaled)
    evaluate_model(y_test, y_pred_dt, "Baseline_DecisionTree", class_names)
    
    print("Saving Baseline model and scaler...")
    os.makedirs('models', exist_ok=True)
    joblib.dump(model_lr, 'models/baseline_lr.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(class_mapping, 'models/class_mapping.pkl')
    print("Done!")

if __name__ == "__main__":
    main()
