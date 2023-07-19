import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import os

os.environ["CUDA_VISIBLE_DEVICES"]="1"

# Veri yüklemesi
data = pd.read_csv("hisse_verileri.csv")

# Veri ön işleme
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data["fiyat"].values.reshape(-1, 1))

# Veri hazırlığı
def prepare_data(data, n_steps):
    X, y = [], []
    for i in range(len(data)-n_steps-1):
        X.append(data[i:i+n_steps, 0])
        y.append(data[i+n_steps, 0])
    return np.array(X), np.array(y)

n_steps = 30  # Her bir veri noktası için kullanılacak önceki adımların sayısı
X, y = prepare_data(scaled_data, n_steps)

# Girdi verilerinin yeniden şekillendirilmesi
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Model oluşturma
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(n_steps, 1)))
model.add(LSTM(128, return_sequences=False))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Modeli eğitme
model.fit(X, y, epochs=11000, batch_size=64)

# Gelecekteki değerleri tahmin etme
future_steps = 10  # Tahmin etmek istediğiniz adım sayısı
last_sequence = scaled_data[-n_steps:]  # Son n_steps veri noktasını alın

predicted_values = []
for _ in range(future_steps):
    input_data = np.reshape(last_sequence, (1, n_steps, 1))
    predicted_value = model.predict(input_data)[0][0]
    predicted_values.append(predicted_value)

    # Son tahmin edilen değeri kullanarak yeni girdi verisi oluşturma
    last_sequence = np.roll(last_sequence, -1)
    last_sequence[-1] = predicted_value

# Tahmin edilen değerleri ters ölçeklendirme
predicted_values = np.array(predicted_values).reshape(-1, 1)
predicted_values = scaler.inverse_transform(predicted_values)

# Tahmin edilen değerleri yazdırma
for i, predicted_value in enumerate(predicted_values):
    print(f"Gelecek {i+1}. adım tahmini: {predicted_value[0]}")

# Tahmin edilen değerlerin ve gerçek verinin bir çizgi grafiğini çizme
plt.figure(figsize=(10, 6))
plt.plot(range(100), scaler.inverse_transform(scaled_data)[-100:], color='blue', label='Gerçek Hisse Senedi Fiyatı')  # Son 100 gün
plt.plot(range(99, 101), [scaler.inverse_transform(scaled_data)[-1], predicted_values[0]], color='red')  # Mavi ve kırmızı çizgi arasındaki bağlantı
plt.plot(range(100, 100 + future_steps), predicted_values, color='red', label='Tahmin Edilen Fiyat')
plt.title('Hisse Senedi Fiyat Tahmini')
plt.xlabel('Adım')
plt.ylabel('Hisse Senedi Fiyatı')
plt.legend()
plt.show()

time.sleep(9999)