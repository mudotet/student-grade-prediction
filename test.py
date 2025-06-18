import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Cấu hình ---
data_path = 'C:/Users/TUS/Downloads/student_grade_prediction/backend/student-mat.csv'
model_path = 'trained_model.npz'

# --- Tiền xử lý dữ liệu ---
def preprocess_data(data):
    data['internet'] = data['internet'].map({'no': 0, 'yes': 1})
    data['famsup'] = data['famsup'].map({'no': 0, 'yes': 1})
    data['schoolsup'] = data['schoolsup'].map({'no': 0, 'yes': 1})

    data = data.dropna(subset=['studytime', 'failures', 'famrel', 'health', 'internet',
                               'absences', 'famsup', 'schoolsup', 'G1', 'G2', 'G3'])

    input_features = ['studytime', 'failures', 'famrel', 'health', 'internet',
                      'absences', 'famsup', 'schoolsup', 'G1', 'G2']
    target_feature = 'G3'

    X = data[input_features].values
    y = data[target_feature].values
    return data, X, y

# --- Hàm dự đoán ---
def predict(X, w, b):
    return np.dot(X, w) + b

# --- Tính MSE ---
def mean_squared_error_manual(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# --- Vẽ biểu đồ ---
def visualize(data, X, y, w, b, cost_history):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_pred = np.array([predict(x, w, b) for x in X_test])
    y_pred = np.clip(y_pred, 0, 20)
    mse = mean_squared_error_manual(y_test, y_pred)

    # --- 1. Vẽ biểu đồ từng thuộc tính đầu vào với G3 ---
    input_features = ['G1', 'G2']
    
    for feature in input_features:
        plt.figure(figsize=(6, 4))
        sns.regplot(x=data[feature], y=data['G3'], ci=None, scatter_kws={'alpha': 0.6})
        plt.title(f"{feature} vs G3 (kèm đường hồi quy)")
        plt.xlabel(feature)
        plt.ylabel("G3")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # --- 2. Đồ thị chi phí ---
    plt.figure(figsize=(8, 5))
    plt.plot(range(len(cost_history)), cost_history, color='blue')
    plt.title("📉 Đồ thị Chi phí theo số lần lặp")
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- 3. So sánh G3 thực tế vs G3 dự đoán ---
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, y_pred, alpha=0.7, color='green')
    plt.plot([0, 20], [0, 20], 'r--')
    plt.title(f"G3 Thực tế vs Dự đoán (MSE = {mse:.2f})")
    plt.xlabel("G3 Thực tế")
    plt.ylabel("G3 Dự đoán")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_pred = np.array([predict(x, w, b) for x in X_test])
    y_pred = np.clip(y_pred, 0, 20)
    mse = mean_squared_error_manual(y_test, y_pred)

    # 1. Ma trận tương quan
    plt.figure(figsize=(12, 8))
    sns.heatmap(data.select_dtypes(include=[np.number]).corr(), annot=True, cmap='coolwarm', fmt=".2f")

    plt.title("📊 Ma trận tương quan giữa các thuộc tính")
    plt.tight_layout()
    plt.show()

# --- Main ---
def main():
    # Đọc dữ liệu
    raw_data = pd.read_csv(data_path)
    data, X, y = preprocess_data(raw_data)

    # Load mô hình đã huấn luyện
    model = np.load(model_path)
    w = model['w']
    b = model['b']
    cost_history = model['cost_history']

    # Hiển thị
    visualize(data, X, y, w, b, cost_history)

# --- Chạy chương trình ---
if __name__ == '__main__':
    main()
