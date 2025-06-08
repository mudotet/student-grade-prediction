from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

app = Flask(__name__)

# Sử dụng CORS để cho phép yêu cầu từ frontend (React)
CORS(app)

# Tải mô hình đã huấn luyện
model_data = np.load('C:/Users/TUS/Downloads/student_grade_prediction/trained_model.npz')
w = model_data['w']
b = model_data['b']

# Hàm dự đoán sử dụng mô hình hồi quy tuyến tính
def predict(features, w, b):
    return np.dot(features, w) + b

@app.route('/predict', methods=['POST'])
def predict_api():
    try:
        # Nhận dữ liệu JSON từ request
        data = request.get_json()

        # Kiểm tra xem 'features' có tồn tại trong dữ liệu hay không
        if 'features' not in data:
            return jsonify({"error": "'features' not found in the request data"}), 400

        # Lấy các đặc trưng từ dữ liệu
        features = data['features']
        
        # Kiểm tra sự đầy đủ của các đặc trưng
        required_features = ['studytime', 'failures', 'famrel', 'health', 'internet', 'absences', 'famsup', 'schoolsup', 'G1', 'G2']
        if len(features) != len(required_features):
            return jsonify({"error": f"Incorrect number of features. Expected {len(required_features)}, but got {len(features)}."}), 400

        # Mã hóa các giá trị phân loại 'internet', 'famsup', 'schoolsup'
        internet = features[4]  # 'internet' (yes or no)
        famsup = features[6]  # 'famsup' (yes or no)
        schoolsup = features[7]  # 'schoolsup' (yes or no)

        # Chuyển các giá trị phân loại thành số (1 hoặc 0)
        features[4] = 1 if internet == 'yes' else 0
        features[6] = 1 if famsup == 'yes' else 0
        features[7] = 1 if schoolsup == 'yes' else 0

        # Chuyển đổi các đặc trưng thành numpy array và chuẩn bị cho mô hình
        features = np.array(features).reshape(1, -1)
        
        # Dự đoán điểm số cuối kỳ (G3)
        prediction = predict(features, w, b)

        # Trả kết quả dự đoán về dưới dạng JSON
        return jsonify({'prediction': prediction[0]})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Khởi động ứng dụng Flask
    app.run(debug=True)
