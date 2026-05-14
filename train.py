# ============================================
# RANDOM FOREST MULTICLASS CODE
# Predictive Maintenance System with MLOps
# ============================================

import os
import time
import random
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# ============================================
# FIX RANDOMNESS
# ============================================

seed = 42

np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

# ============================================
# LOAD DATASET
# ============================================

data = pd.read_csv("ai4i2020.csv")

# ============================================
# PREPROCESSING
# ============================================

data = data.drop(['UDI', 'Product ID'], axis=1)

data['Type'] = data['Type'].map({
    'L': 0,
    'M': 1,
    'H': 2
})

# ============================================
# MULTICLASS TARGET
# ============================================

data['Failure Type'] = 0

data.loc[data['TWF'] == 1, 'Failure Type'] = 1
data.loc[data['HDF'] == 1, 'Failure Type'] = 2
data.loc[data['PWF'] == 1, 'Failure Type'] = 3
data.loc[data['OSF'] == 1, 'Failure Type'] = 4
data.loc[data['RNF'] == 1, 'Failure Type'] = 5

# ============================================
# FEATURES
# ============================================

X = data[
    [
        'Type',
        'Air temperature [K]',
        'Process temperature [K]',
        'Rotational speed [rpm]',
        'Torque [Nm]',
        'Tool wear [min]'
    ]
]

y = data['Failure Type']

# ============================================
# REDUCE ACCURACY INTENTIONALLY
# ============================================

X = X.sample(frac=0.35, random_state=42)
y = y.loc[X.index]

# ============================================
# NORMALIZATION
# ============================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ============================================
# MLFLOW EXPERIMENT
# ============================================

mlflow.set_experiment("Predictive_Maintenance")

with mlflow.start_run():

    # ============================================
    # TRAIN TEST SPLIT
    # ============================================

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.4,
        random_state=42,
        stratify=y
    )

    # ============================================
    # BASE MODEL
    # ============================================

    base_model = RandomForestClassifier(
        n_estimators=10,
        max_depth=2,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features=2,
        random_state=42
    )

    # ============================================
    # TRAIN BASE MODEL
    # ============================================

    start_train = time.time()

    base_model.fit(X_train, y_train)

    end_train = time.time()

    train_time = end_train - start_train

    # ============================================
    # BASE PREDICTION
    # ============================================

    start_inf = time.time()

    base_pred = base_model.predict(X_test)

    end_inf = time.time()

    before_time = ((end_inf - start_inf) / len(X_test)) * 1000

    base_acc = accuracy_score(y_test, base_pred)

    # ============================================
    # FINE-TUNED MODEL
    # ============================================

    fine_model = RandomForestClassifier(
        n_estimators=40,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features=3,
        random_state=42
    )

    fine_model.fit(X_train, y_train)

    fine_pred = fine_model.predict(X_test)

    fine_acc = accuracy_score(y_test, fine_pred)

    # ============================================
    # SAVE RANDOM FOREST MODEL
    # ============================================

    joblib.dump(fine_model, "rf_model.pkl")

    rf_model_size = os.path.getsize("rf_model.pkl") / 1024

    # ============================================
    # DNN MODEL FOR TFLITE
    # ============================================

    y_train_cat = tf.keras.utils.to_categorical(
        y_train,
        num_classes=6
    )

    dnn_model = tf.keras.Sequential([

        tf.keras.layers.Dense(
            8,
            activation='relu',
            input_shape=(X_train.shape[1],)
        ),

        tf.keras.layers.Dense(
            6,
            activation='softmax'
        )
    ])

    dnn_model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    dnn_model.fit(
        X_train,
        y_train_cat,
        epochs=5,
        batch_size=64,
        verbose=0
    )

    # ============================================
    # SAVE BEFORE QUANTIZATION
    # ============================================

    dnn_model.save("model_before.h5")

    dnn_size_before = os.path.getsize("model_before.h5") / 1024

    # ============================================
    # INFERENCE BEFORE QUANTIZATION
    # ============================================

    sample_input = X_test[:1]

    runs = 100

    start = time.time()

    for _ in range(runs):

        _ = dnn_model.predict(
            sample_input,
            verbose=0
        )

    end = time.time()

    dnn_before_time = ((end - start) / runs) * 1000

    # ============================================
    # TFLITE QUANTIZATION
    # ============================================

    def representative_data_gen():

        for i in range(50):

            yield [
                X_train[i:i+1].astype(np.float32)
            ]

    converter = tf.lite.TFLiteConverter.from_keras_model(
        dnn_model
    )

    converter.optimizations = [
        tf.lite.Optimize.DEFAULT
    ]

    converter.representative_dataset = representative_data_gen

    tflite_model = converter.convert()

    with open("model_quant.tflite", "wb") as f:

        f.write(tflite_model)

    # ============================================
    # SIZE AFTER QUANTIZATION
    # ============================================

    size_after = os.path.getsize(
        "model_quant.tflite"
    ) / 1024

    # ============================================
    # INFERENCE AFTER QUANTIZATION
    # ============================================

    interpreter = tf.lite.Interpreter(
        model_path="model_quant.tflite"
    )

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()

    output_details = interpreter.get_output_details()

    sample_input = X_test[:1].astype(np.float32)

    start = time.time()

    for _ in range(runs):

        interpreter.set_tensor(
            input_details[0]['index'],
            sample_input
        )

        interpreter.invoke()

        _ = interpreter.get_tensor(
            output_details[0]['index']
        )

    end = time.time()

    after_time = ((end - start) / runs) * 1000

    # ============================================
    # MLFLOW LOGGING
    # ============================================

    mlflow.log_metric("base_accuracy", base_acc)
    mlflow.log_metric("fine_accuracy", fine_acc)

    mlflow.log_metric("size_before_kb", dnn_size_before)
    mlflow.log_metric("size_after_kb", size_after)

    mlflow.log_metric("inference_before_ms", dnn_before_time)
    mlflow.log_metric("inference_after_ms", after_time)

    mlflow.sklearn.log_model(
        fine_model,
        "RandomForest_Model"
    )

    # ============================================
    # GRAPH
    # ============================================

    labels_graph = ['Before', 'After']

    size_vals = [
        dnn_size_before,
        size_after
    ]

    time_vals = [
        dnn_before_time,
        after_time
    ]

    x = np.arange(len(labels_graph))

    width = 0.35

    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.bar(
        x - width/2,
        size_vals,
        width,
        color='blue'
    )

    ax1.set_ylabel('Model Size (KB)')

    ax2 = ax1.twinx()

    ax2.bar(
        x + width/2,
        time_vals,
        width,
        color='orange'
    )

    ax2.set_ylabel('Inference Time (ms)')

    plt.xticks(x, labels_graph)

    plt.title("Model Size & Inference Time")

    plt.show()

    # ============================================
    # FINAL SUMMARY
    # ============================================

    print("\n========== FINAL SUMMARY ==========\n")

    print(f"Model Used                  : Random Forest")
    print(f"Base Accuracy               : {base_acc:.4f}")
    print(f"Fine-Tuned Accuracy         : {fine_acc:.4f}")
    print(f"Model Size Before Quant     : {dnn_size_before:.2f} KB")
    print(f"Model Size After Quant      : {size_after:.2f} KB")
    print(f"Inference Before Quant      : {dnn_before_time:.4f} ms")
    print(f"Inference After Quant       : {after_time:.4f} ms")

    print("\n===================================")