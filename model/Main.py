from DataLoader import loaddata
from Model import build_model
from Train import train_model
from keras.preprocessing.image import ImageDataGenerator

train_dir = 'model/working/cats-v-dogs/training'
val_dir = 'model/working/cats-v-dogs/validation'
test_dir = 'model/working/cats-v-dogs/test'

train_generator, validation_generator, test_ds = loaddata(train_dir, val_dir, test_dir)

model = build_model()

history, evaluation = train_model(model, train_generator, validation_generator, test_ds)

# Access the relevant information
print(history.history)  # Access training history
print("[+] Result:")

for name, value in evaluation.items():
    print(f"{name}: {value:.4f}")

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    directory='/content/working/cats-v-dogs/test/',
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary',
    shuffle=False  # Set to False to keep the order of predictions
)

# Assuming your model predicts probabilities
y_pred_prob = model.predict(test_generator)
y_pred = (y_pred_prob > 0.5).astype(int)

# Get true labels
y_true = test_generator.classes

# Compute precision, recall, and F1-score
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')

print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-Score: {f1:.4f}')
