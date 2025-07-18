import glob
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from PIL import Image
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")

# Görsellerin yolu
imgs = glob.glob("./img_nihai/*.png")

width = 125
height = 50

X = []
Y = []

# Görselleri yükle, normalize et ve label çıkar
for img in imgs:
    filename = os.path.basename(img)
    label = filename.split("_")[0]  # Dosya adı başına göre label
    im = np.array(Image.open(img).convert("L").resize((width, height)))
    im = im / 255.0  # Normalizasyon
    X.append(im)
    Y.append(label)

X = np.array(X)
# TensorFlow için 4 boyutlu tensör: (num_samples, height, width, channels)
X = X.reshape(X.shape[0], height, width, 1)

def onehot_labels(values):
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    onehot_encoder = OneHotEncoder(sparse_output=False)  # Burada değişiklik yapıldı
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return onehot_encoded

Y = onehot_labels(Y)

train_X, test_X, train_y, test_y = train_test_split(X, Y, test_size=0.25, random_state=2)

# CNN Modeli
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(height, width, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.4))
model.add(Dense(Y.shape[1], activation="softmax"))  # Output katmanını otomatik label sayısına göre ayarladım

# Modeli derle
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Modeli eğit
model.fit(train_X, train_y, epochs=35, batch_size=64)

# Eğitim doğruluğu
score_train = model.evaluate(train_X, train_y)
print("Eğitim doğruluğu: %", score_train[1] * 100)

# Test doğruluğu
score_test = model.evaluate(test_X, test_y)
print("Test doğruluğu: %", score_test[1] * 100)

# Model mimarisi ve ağırlıkları kaydet
with open("model_new.json", "w") as json_file:
    json_file.write(model.to_json())

model.save_weights("trex_weight_new.weights.h5")

