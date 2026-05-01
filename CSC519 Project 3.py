import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report


# CSC519 – Handwritten Digit Classification (MLP & CNN)


print("=" * 60)
print("TASK 1: Data Preparation")
print("=" * 60)

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train / 255.0
X_test = X_test / 255.0

y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples    : {X_test.shape[0]}\n")

# Save sample images
fig, axes = plt.subplots(1, 10, figsize=(15, 2))
for i in range(10):
    axes[i].imshow(X_train[i], cmap='gray')
    axes[i].set_title(f"Label: {y_train[i]}")
    axes[i].axis('off')
plt.suptitle("Sample MNIST Digits")
plt.tight_layout()
plt.savefig("sample_images.png", dpi=150)
plt.close()
print("Saved: sample_images.png\n")

# TASK 2: Multilayer Perceptron (MLP)

print("=" * 60)
print("TASK 2: Multilayer Perceptron")
print("=" * 60)

mlp_model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
], name="MLP")

mlp_model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

mlp_model.summary()

start = time.time()
mlp_history = mlp_model.fit(X_train, y_train_cat,
                            epochs=10,
                            batch_size=128,
                            validation_split=0.1,
                            verbose=1)
mlp_time = time.time() - start

mlp_loss, mlp_acc = mlp_model.evaluate(X_test, y_test_cat, verbose=0)

print(f"MLP Test Accuracy : {mlp_acc*100:.2f}%")
print(f"MLP Training Time : {mlp_time:.1f}s\n")

# Plot MLP curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(mlp_history.history['accuracy'], label='Train')
ax1.plot(mlp_history.history['val_accuracy'], label='Val')
ax1.set_title('MLP Accuracy')
ax1.legend()

ax2.plot(mlp_history.history['loss'], label='Train')
ax2.plot(mlp_history.history['val_loss'], label='Val')
ax2.set_title('MLP Loss')
ax2.legend()

plt.suptitle("MLP Training Curves")
plt.tight_layout()
plt.savefig("mlp_training_curves.png", dpi=150)
plt.close()
print("Saved: mlp_training_curves.png\n")



# TASK 3: Convolutional Neural Network (CNN)
print("=" * 60)
print("TASK 3: Convolutional Neural Network")
print("=" * 60)

X_train_cnn = X_train.reshape(-1, 28, 28, 1)
X_test_cnn = X_test.reshape(-1, 28, 28, 1)

cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
], name="CNN")

cnn_model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

cnn_model.summary()

start = time.time()
cnn_history = cnn_model.fit(X_train_cnn, y_train_cat,
                            epochs=10,
                            batch_size=128,
                            validation_split=0.1,
                            verbose=1)
cnn_time = time.time() - start

cnn_loss, cnn_acc = cnn_model.evaluate(X_test_cnn, y_test_cat, verbose=0)

print(f"CNN Test Accuracy : {cnn_acc*100:.2f}%")
print(f"CNN Training Time : {cnn_time:.1f}s\n")

# Plot CNN curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(cnn_history.history['accuracy'], label='Train')
ax1.plot(cnn_history.history['val_accuracy'], label='Val')
ax1.set_title('CNN Accuracy')
ax1.legend()

ax2.plot(cnn_history.history['loss'], label='Train')
ax2.plot(cnn_history.history['val_loss'], label='Val')
ax2.set_title('CNN Loss')
ax2.legend()

plt.suptitle("CNN Training Curves")
plt.tight_layout()
plt.savefig("cnn_training_curves.png", dpi=150)
plt.close()
print("Saved: cnn_training_curves.png\n")


# TASK 4: Model Comparison
print("=" * 60)
print("TASK 4: Model Comparison")
print("=" * 60)

print(f"{'Metric':<18} {'MLP':>10} {'CNN':>10}")
print("-" * 42)
print(f"{'Accuracy (%)':<18} {mlp_acc*100:>10.2f} {cnn_acc*100:>10.2f}")
print(f"{'Loss':<18} {mlp_loss:>10.4f} {cnn_loss:>10.4f}")
print(f"{'Time (s)':<18} {mlp_time:>10.1f} {cnn_time:>10.1f}")
print("-" * 42)

# Comparison bar chart
categories = ['Accuracy (%)', 'Loss']
mlp_vals = [mlp_acc*100, mlp_loss]
cnn_vals = [cnn_acc*100, cnn_loss]
x = np.arange(len(categories))

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - 0.17, mlp_vals, 0.34, label='MLP', color='steelblue')
ax.bar(x + 0.17, cnn_vals, 0.34, label='CNN', color='coral')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()
ax.set_title("MLP vs CNN Performance")

plt.tight_layout()
plt.savefig("comparison_chart.png", dpi=150)
plt.close()
print("Saved: comparison_chart.png\n")


# TASK 5: Optimized CNN
print("=" * 60)
print("TASK 5: Optimized CNN")
print("=" * 60)

opt_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(2, 2),
    Dropout(0.25),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
], name="Optimized_CNN")

opt_model.compile(optimizer=Adam(learning_rate=0.0005),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

opt_model.summary()

start = time.time()
opt_history = opt_model.fit(X_train_cnn, y_train_cat,
                            epochs=10,
                            batch_size=128,
                            validation_split=0.1,
                            verbose=1)
opt_time = time.time() - start

opt_loss, opt_acc = opt_model.evaluate(X_test_cnn, y_test_cat, verbose=0)

print(f"Optimized CNN Accuracy : {opt_acc*100:.2f}%")
print(f"Training Time          : {opt_time:.1f}s\n")

# Final summary
print("=" * 60)
print("FINAL RESULTS")
print("=" * 60)
print(f"{'Model':<18} {'Accuracy':>9} {'Loss':>10} {'Time(s)':>8}")
print("-" * 50)
print(f"{'MLP':<18} {mlp_acc*100:>8.2f}% {mlp_loss:>10.4f} {mlp_time:>8.1f}")
print(f"{'CNN':<18} {cnn_acc*100:>8.2f}% {cnn_loss:>10.4f} {cnn_time:>8.1f}")
print(f"{'Optimized CNN':<18} {opt_acc*100:>8.2f}% {opt_loss:>10.4f} {opt_time:>8.1f}")
print("-" * 50)

# Classification report
y_pred = np.argmax(opt_model.predict(X_test_cnn, verbose=0), axis=1)
print("\nClassification Report (Optimized CNN):")
print(classification_report(y_test, y_pred,
                            target_names=[f"Digit {i}" for i in range(10)]))

print("\nFigures saved:")
print(" - sample_images.png")
print(" - mlp_training_curves.png")
print(" - cnn_training_curves.png")
print(" - comparison_chart.png")
print(" - optimized_cnn_curves.png")