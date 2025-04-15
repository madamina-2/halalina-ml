from flask import Flask, request, jsonify
import pandas as pd
import pickle
from flask_jwt_extended import JWTManager, jwt_required
import os

# Inisialisasi Flask dan JWT
app = Flask(__name__)
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'your-jwt-secret-key')  # Ganti dengan kunci rahasia yang lebih aman
jwt = JWTManager(app)

# Load model terbaik (pipeline lengkap) dari file
with open('model/best_random_forest_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

@app.route('/predict', methods=['POST'])
@jwt_required()  # Hanya bisa diakses jika ada JWT yang valid
def predict():
    try:
        # Ambil data dari request JSON
        data = request.get_json()

        # Membuat DataFrame baru sesuai dengan inputan
        new_data = pd.DataFrame([{
            'job': data['job'],
            'marital': data['marital'],
            'balance': data['balance'],
            'age_group': data['age_group'],
            'is_having_debt': data['is_having_debt']
        }])

        # Lakukan prediksi
        prediction = loaded_model.predict(new_data)

        # Jika ingin probabilitas per kelas
        proba = loaded_model.predict_proba(new_data)

        # Kembalikan response dalam bentuk JSON
        response = {
            "prediction": prediction.tolist(),
            "confidence_scores": proba.tolist()
        }
        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
