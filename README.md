## CIFAR-10 Image Classification using TensorFlow

This project focuses on building an end-to-end image classification pipeline using the CIFAR-10 dataset. It includes data loading, preprocessing, visualization, and training a simple Convolutional Neural Network (CNN) model using TensorFlow and Keras.

---

## Project Overview

The CIFAR-10 dataset consists of 60,000 32×32 color images across 10 different classes such as airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

In this project, the following steps are implemented:

* Loading the CIFAR-10 dataset
* Data preprocessing and normalization
* Exploratory data analysis and visualization
* Building a simple CNN model
* Training and validating the model
* Evaluating performance on test data

---

## Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Matplotlib
* Jupyter Notebook

---

## Project Structure

```
.
├── cifar10_data_analysis_tensorflow.ipynb
├── README.md
└── requirements.txt (optional)
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

Install dependencies:

```bash
pip install tensorflow numpy matplotlib
```

---

## How to Run

Open the notebook:

```bash
jupyter notebook cifar10_data_analysis_tensorflow.ipynb
```

Run all cells to:

1. Load and preprocess the dataset
2. Visualize samples
3. Train the CNN model
4. Evaluate accuracy

---

## Model Used

A simple Convolutional Neural Network (CNN):

* Conv2D + ReLU
* MaxPooling
* Flatten
* Dense layers
* Softmax output for 10 classes

Loss function: `sparse_categorical_crossentropy`
Optimizer: `Adam`

---

## Results

The model is trained and evaluated on the CIFAR-10 test set. Accuracy and loss graphs are plotted to analyze performance.

---

## Future Improvements

* Add data augmentation
* Build a deeper CNN architecture
* Apply Dropout and Batch Normalization
* Hyperparameter tuning
* Deploy using Streamlit

