import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Function to predict values using linear regression (dot product)
def predict(x, w, b):
    p = np.dot(x, w) + b
    return p

# Function to compute the cost
def compute_cost(X, y, w, b): 
    m = X.shape[0]
    cost = 0.0
    for i in range(m):                                  
        f_wb_i = np.dot(X[i], w) + b                    
        cost += (f_wb_i - y[i])**2                     
    cost = cost / (2 * m)                                
    return cost

# Function to compute the gradient
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

# Function for gradient descent
def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
    J_history = []   
    w = np.copy(w_in)  # Avoid modifying global w
    b = b_in
    
    for i in range(num_iters):
        dj_db, dj_dw = gradient_function(X, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        
        if i < 100000:  # Prevent resource exhaustion
            J_history.append(cost_function(X, y, w, b))

        if i % 100 == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}")
        
    return w, b, J_history

# Function to preprocess the data
def preprocess_data(data):
    # Mã hóa nhị phân
    data['internet'] = data['internet'].map({'no': 0, 'yes': 1})
    data['famsup'] = data['famsup'].map({'no': 0, 'yes': 1})
    data['schoolsup'] = data['schoolsup'].map({'no': 0, 'yes': 1})

    # studytime giữ nguyên (1-4)
    # failures giữ nguyên (0-3)
    # Kiểm tra và xử lý giá trị ngoài khoảng
    data['failures'] = data['failures'].clip(0, 3)

    # Kiểm tra NaN
    data = data.dropna(subset=['studytime', 'failures', 'famrel', 'health', 'internet', 'absences', 'famsup', 'schoolsup', 'G1', 'G2', 'G3'])

    input_features = ['studytime', 'failures', 'famrel', 'health', 'internet', 'absences', 'famsup', 'schoolsup', 'G1', 'G2']
    target_feature = 'G3'

    X = data[input_features].values
    y = data[target_feature].values

    # Chuẩn hóa dữ liệu đầu vào
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y


# Function to train the model
def train_model():
    # Load dataset
    file_path = 'C:/Users/TUS/Downloads/student_grade_prediction/backend/student-mat.csv'
    data = pd.read_csv(file_path)
    
    # Preprocess the data
    X, y = preprocess_data(data)

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize parameters
    w_init = np.random.randn(X_train.shape[1]) * 0.01
    b_init = np.random.randn() * 0.01

    # Gradient descent settings
    alpha = 0.1
    iterations = 10000

    # Run gradient descent
    w_final, b_final, J_hist = gradient_descent(X_train, y_train, w_init, b_init,
                                                 compute_cost, compute_gradient,
                                                 alpha, iterations)
    
    print(f"Final w: {w_final}, b: {b_final}")
    
    # Evaluate the model on the test set
    y_pred = np.array([predict(x, w_final, b_final) for x in X_test])
    y_pred = np.clip(y_pred, 0, 20)  # Đảm bảo dự đoán nằm trong khoảng 0-20
    
    # Calculate Mean Squared Error (MSE) for the test set
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error on Test Set: {mse}')
    
    # Save the learned parameters and cost history to a file
    np.savez('trained_model.npz', w=w_final, b=b_final, cost_history=J_hist)
    print("Model and cost history saved to 'trained_model.npz'.")

# Run the model training
train_model()
