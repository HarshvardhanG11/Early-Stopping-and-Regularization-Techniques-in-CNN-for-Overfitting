![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-DeepLearning-red?logo=keras)
![Matplotlib](https://img.shields.io/badge/Matplotlib-DataViz-yellow?logo=plotly)
![License](https://img.shields.io/badge/License-MIT-green)
![Stars](https://img.shields.io/github/stars/yourusername/yourrepo?style=social)
<img src="https://img.icons8.com/color/48/000000/artificial-intelligence.png" width="40"/>
<img src="https://img.icons8.com/color/48/000000/computer.png" width="40"/>

# 📌 Early Stopping and Regularization Techniques in CNN for Overfitting

🚀 This project demonstrates how Convolutional Neural Networks (CNNs) can be improved using Early Stopping and Regularization techniques to overcome overfitting while training on the MNIST dataset.

---

## 🌟 Features

* ✅ Handwritten Digit Recognition using **MNIST Dataset (0–9 digits)**
* ✅ **L2 Regularization** to control large weights
* ✅ **Dropout Layers** to reduce dependency on neurons
* ✅ **Batch Normalization** to stabilize training
* ✅ **Early Stopping** to stop training at the right time
* ✅ Achieves **high accuracy (\~98%)** with reduced overfitting

---

## 🛠️ Tech Stack

### 💻 Languages Used

* 🐍 **Python 3**

### 📚 Libraries & Tools

* 🔹 **TensorFlow / Keras** → Deep Learning Framework
* 🔹 **Matplotlib** → Plotting Loss & Accuracy curves
* 🔹 **NumPy** → Numerical Operations

---

## ⚙️ How It Works

1. **Load Dataset** → MNIST handwritten digits.
2. **Preprocess Data** → Normalize images to range `[0,1]`.
3. **Build CNN Model**

   * Conv2D → Feature Extraction
   * Batch Normalization → Stabilization
   * MaxPooling2D → Downsampling
   * Dropout → Regularization
   * Dense Layers → Classification
4. **Add Regularization**

   * L2 penalty on convolutional layers
   * Dropout between layers
5. **Train with Early Stopping** → Stops training when validation loss stops improving.
6. **Evaluate** → Test dataset for final accuracy.
7. **Visualize** → Training vs Validation curves.

---

## 📊 Results

* 📈 **Training Accuracy**: > 99%
* 📉 **Validation Accuracy**: \~ 98%
* ✅ **Overfitting minimized** using Dropout, L2, and Early Stopping

---

## 🧩 Pseudo Code

```python
# Pseudo-code for CNN with Early Stopping & Regularization

load MNIST dataset
normalize images to [0,1]

define CNN model:
    Conv2D + ReLU + L2 regularization
    Batch Normalization
    MaxPooling
    Dropout
    
    Conv2D + ReLU + L2 regularization
    Batch Normalization
    MaxPooling
    Dropout

    Flatten
    Dense Layer (128) + ReLU
    Dropout
    Dense Layer (10) + Softmax

compile model with Adam optimizer + cross-entropy loss

set EarlyStopping(monitor="val_loss", patience=3)

train model on training data with validation split

evaluate on test dataset

plot training loss vs validation loss
```

---

## 📷 Demo (Architecture Flow)

👉 CNN Layers → Batch Normalization → Dropout → Early Stopping → Dense Softmax

---

## 🤝 Contributors

👨‍💻 Made by **Contrubution of team members** with ❤️

---

---

