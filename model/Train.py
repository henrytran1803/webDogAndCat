# Trainer.py
from tensorflow import keras
import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.callbacks import TensorBoard

def train_model(model, train_generator, validation_generator, test_ds):
    tensorboard_callback = TensorBoard(log_dir='/content/logs', histogram_freq=1)

    callbacks = [early_stopping_callback, tensorboard_callback]

    # Huấn luyện mô hình với callbacks
    history = model.fit(train_generator, epochs=15, validation_data=validation_generator, callbacks=callbacks)

    evaluation = model.evaluate(test_ds, return_dict=True)

    print("[+] Result:")

    for name, value in evaluation.items():
        print(f"{name}: {value:.4f}")

    # Return relevant information
    return history, evaluation
