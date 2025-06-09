import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Hàm dự đoán giá trị sử dụng hồi quy tuyến tính (tích vô hướng)
def predict(x, w, b):
    p = np.dot(x, w) + b
    return p

# Hàm tính toán chi phí (cost)
def compute_cost(X, y, w, b): 
    m = X.shape[0]
    cost = 0.0
    for i in range(m):                                  
        f_wb_i = np.dot(X[i], w) + b                    
        cost += (f_wb_i - y[i])**2                     
    cost = cost / (2 * m)                                
    return cost

# Hàm tính gradient
def compute_gradient(X, y, w, b): 
    m, n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.0

    for i in range(m):                                 
        err = (np.dot(X[i], w) + b) - y[i]             
        for j in range(n):                             
            dj_dw[j] += err * X[i, j]                 
        dj_db += err                                   
    dj_dw = dj_dw / m                                  
    dj_db = dj_db / m                                  
    
    return dj_db, dj_dw

# Hàm thực hiện gradient descent
def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
    J_history = []   
    w = np.copy(w_in)  # Tránh thay đổi w toàn cục
    b = b_in
    
    for i in range(num_iters):
        dj_db, dj_dw = gradient_function(X, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        
        if i < 100000:  # Ngăn ngừa tràn bộ nhớ
            J_history.append(cost_function(X, y, w, b))

        if i % 100 == 0:
            print(f"Lặp {i:4d}: Chi phí {J_history[-1]:8.2f}")
        
    return w, b, J_history

def mean_squared_error_manual(y_true, y_pred):
    """
    Tính toán Mean Squared Error (MSE) giữa y_true và y_pred.
    """
    n = len(y_true)
    mse = np.sum((y_true - y_pred) ** 2) / n
    return mse

# Hàm tiền xử lý dữ liệu
def preprocess_data(data):
    # Mã hóa nhị phân cho các cột dạng yes/no
    data['internet'] = data['internet'].map({'no': 0, 'yes': 1})
    data['famsup'] = data['famsup'].map({'no': 0, 'yes': 1})
    data['schoolsup'] = data['schoolsup'].map({'no': 0, 'yes': 1})

    # Giới hạn giá trị cột 'failures' trong khoảng 0-3
    data['failures'] = data['failures'].clip(0, 3)

    # Kiểm tra và loại bỏ các dòng có giá trị NaN (không phải số)
    data = data.dropna(subset=['studytime', 'failures', 'famrel', 'health', 'internet', 'absences', 'famsup', 'schoolsup', 'G1', 'G2', 'G3'])

    input_features = ['studytime', 'failures', 'famrel', 'health', 'internet', 'absences', 'famsup', 'schoolsup', 'G1', 'G2']
    target_feature = 'G3'

    X = data[input_features].values
    y = data[target_feature].values

    return X, y

# Hàm huấn luyện mô hình
def train_model():
    # Đọc dữ liệu từ file
    file_path = 'C:/Users/TUS/Downloads/student_grade_prediction/backend/student-mat.csv'
    data = pd.read_csv(file_path)
    # Tiền xử lý dữ liệu
    X, y = preprocess_data(data)
    print(X)
    print(y)

    # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Khởi tạo tham số ban đầu
    w_init = np.random.randn(X_train.shape[1]) * 0.01
    b_init = np.random.randn() * 0.01

    # Thiết lập tham số cho gradient descent
    alpha = 0.003
    iterations = 10000

    # Thực hiện gradient descent
    w_final, b_final, J_hist = gradient_descent(X_train, y_train, w_init, b_init,
                                                 compute_cost, compute_gradient,
                                                 alpha, iterations)
    
    print(f"Trọng số cuối cùng w: {w_final}, b: {b_final}")
    
    # Đánh giá mô hình trên tập kiểm tra
    y_pred = np.array([predict(x, w_final, b_final) for x in X_test])
    y_pred = np.clip(y_pred, 0, 20)  # Đảm bảo dự đoán nằm trong khoảng 0-20
    
    # Tính toán Mean Squared Error (MSE) trên tập kiểm tra
    mse = mean_squared_error_manual(y_test, y_pred)
    print(f'MSE trên tập kiểm tra: {mse}')
    
    # Lưu trọng số và lịch sử chi phí vào file
    np.savez('trained_model.npz', w=w_final, b=b_final, cost_history=J_hist)
    print("Đã lưu mô hình và lịch sử chi phí vào 'trained_model.npz'.")

# Chạy huấn luyện mô hình
train_model()
