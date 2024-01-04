import cv2
import views
from tensorflow import keras
import os
from matplotlib import pyplot as plt
test_img = cv2.imread('streaming/anh-cho-cuoi.jpg')
plt.imshow(test_img)
plt.show()
test_img = cv2.resize(test_img,(256,256))
test_input = test_img.reshape((1,256,256,3))
def load_model():
    model_path = "model/model.h5"

    if os.path.exists(model_path):
        try:
            loaded_model = keras.models.load_model(model_path)
            return loaded_model
        except Exception as e:
            print(f"Error loading the model: {e}")
            return None
    else:
        print(f"Error: Model file '{model_path}' not found.")
        return None

# Usage
model = load_model()


if model is not None:
    predictions = model.predict(test_input)
    print(predictions)
else:
    print('none')