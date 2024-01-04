from keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image




def test_model(model, image_path):
    # Load và chuyển đổi hình ảnh về định dạng mà mô hình có thể xử lý
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Chuẩn hóa giá trị pixel về khoảng [0, 1]

    # Dự đoán lớp của hình ảnh
    prediction = model.predict(img_array)

    if prediction[0, 0] > 0.5:
        result = "Dog"
    else:
        result = "Cat"

    return result, prediction[0, 0]

model3 = load_model('model/model.h5')
image_path_to_test = "streaming/hinh-nen-con-meo-cute-1.jpg"
result, confidence = test_model(model3, image_path_to_test)

print(f"Prediction: {result} with confidence: {confidence}")
