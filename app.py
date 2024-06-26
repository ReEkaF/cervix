from flask import Flask, request, render_template, redirect, url_for
import pickle
import numpy as np

# Inisialisasi aplikasi Flask
app = Flask(__name__)
application = app

# Memuat model yang telah disimpan
with open('gnb_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Route untuk halaman utama (form input data)
@app.route('/')
def home():
    return render_template('index.html')

# Route untuk prediksi
@app.route('/predict', methods=['POST'])
def predict():
    # Mendapatkan data dari form
    features = [int(x) for x in request.form.values()]
    final_features = [np.array(features)]
    
    # Melakukan prediksi menggunakan model yang telah dilatih
    prediction = model.predict(final_features)
    
    # Menginterpretasi hasil prediksi
    output = 'negative' if prediction[0] == 0 else 'Positive'
    
    # Mengarahkan ke halaman hasil prediksi
    return redirect(url_for('result', prediction_text=output, input_data=",".join(map(str, features))))

# Route untuk menampilkan hasil prediksi
@app.route('/result')
def result():
    prediction_text = request.args.get('prediction_text')
    input_data = request.args.get('input_data').split(',')
    input_data = list(map(int, input_data))
    accuracy_model = 1.00
    return render_template('result.html', prediction_text=prediction_text, input_data=input_data, accuracy_model=accuracy_model)

if __name__ == '__main__':
    app.run(debug=True)