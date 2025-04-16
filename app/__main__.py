from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import pickle
from flask_jwt_extended import JWTManager, jwt_required
import os
from dotenv import load_dotenv  # Untuk memuat environment variable dari file .env
from flask_cors import CORS  # Importing CORS

# Memuat .env file
load_dotenv()

# Inisialisasi Flask dan JWT
app = Flask(__name__)
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'your-jwt-secret-key')  # Ganti dengan kunci rahasia yang lebih aman
JWTManager(app)

# Enable CORS for all origins
CORS(app, resources={r"/*": {"origins": ["http://127.0.0.1:5002", "http://192.168.23.169:5002"]}})  # This will allow all origins to access your API

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
        

        # Mapping klaster ke profil risiko
        risk_profiles = {
            0: "Agresif",
            1: "Moderat",
            2: "Defensif"
        }

        # Base model alokasi portofolio (persentase) untuk masing-masing profil
        base_allocations = { 
            "Agresif": {
                "Tabungan Emas": 0.40,
                "SBSN": 0.30,
                "RDPU Syariah": 0.20,
                "Deposito Syariah": 0.10
            },
            "Moderat": {
                "Tabungan Emas": 0.25,
                "SBSN": 0.35,
                "RDPU Syariah": 0.25,
                "Deposito Syariah": 0.15
            },
            "Defensif": {
                "Tabungan Emas": 0.05,
                "SBSN": 0.40,
                "RDPU Syariah": 0.30,
                "Deposito Syariah": 0.25
            }
        }

        # Ambil probabilitas per kelas dari model
        proba_scores = np.array(proba).flatten()

        # Hitung total alokasi berbobot dari semua base model dikalikan proba
        final_allocation = {
            "Tabungan Emas": 0.0,
            "SBSN": 0.0,
            "RDPU Syariah": 0.0,
            "Deposito Syariah": 0.0
        }

        # Bobot: proba[0] = agresif, proba[1] = moderat, proba[2] = defensif
        for i, profile in enumerate(["Agresif", "Moderat", "Defensif"]):
            for product in final_allocation:
                final_allocation[product] += base_allocations[profile][product] * proba_scores[i]

        # Konversi ke persen (float)
        percent_values = {k: v * 100 for k, v in final_allocation.items()}

        # Simpan pembulatan ke bawah dan sisa desimal
        floored = {k: int(v) for k, v in percent_values.items()}
        remainders = {k: percent_values[k] - floored[k] for k in percent_values}

        # Hitung total awal dan selisih
        total_floored = sum(floored.values())
        diff = 100 - total_floored

        # Urutkan berdasarkan sisa desimal terbesar → tambahkan 1 ke elemen tersebut hingga total = 100
        sorted_remainders = sorted(remainders.items(), key=lambda x: x[1], reverse=True)

        # Tambahkan 1 ke elemen yang sisa desimalnya terbesar
        for i in range(diff):
            floored[sorted_remainders[i][0]] += 1

        # Tampilkan hasil
        predicted_cluster = int(prediction[0])
        risk_profile = risk_profiles[predicted_cluster]

        # Kembalikan response dalam bentuk JSON
        response = {
            "risk_profile": risk_profile,
            "floored_percentages": floored 
        }
        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Menjalankan aplikasi
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
