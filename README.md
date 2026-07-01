# 👁️ Eye Disease Classification Using Deep Learning 

## 🚀 Project Overview

This project presents a comprehensive deep learning study for eye disease classification using retinal images. It explores and compares multiple state-of-the-art CNN architectures and transfer learning models to analyze performance, generalization ability, and computational efficiency.

The study focuses not only on classification accuracy but also on:

* Model complexity vs performance trade-off
* Overfitting behavior and mitigation strategies
* Training stability under different hyperparameters
* Efficiency of lightweight vs heavy architectures

## 🧠 Models Implemented

A total of 7 deep learning architectures are evaluated:

**🔹 Custom & Classic CNN**
* Custom CNN (baseline model)

**🔹 Transfer Learning Models**
* VGG16
* ResNet50
* MobileNetV2
* EfficientNetB0
* DenseNet121
* InceptionV3

Each model is trained and evaluated under:

* Baseline configuration
* Improved configuration (regularization, dropout, fine-tuning)
* Scratch vs ImageNet pretrained comparison

## 📊 Dataset
* **Source:** Kaggle – Eye Diseases Classification Dataset
* **Classes:**
  * Cataract
  * Diabetic Retinopathy
  * Glaucoma
  * Normal
* **Preprocessing Pipeline:**
  * Image resizing to 224×224
  * Train/Validation/Test split (80/10/10)
  * Data normalization
  * Advanced augmentation (flip, rotation, zoom, brightness, contrast)

## 🔧 Techniques Used

This project includes advanced deep learning techniques:

**🧪 Data Processing**
* Image augmentation pipeline
* TensorFlow dataset optimization (cache & prefetch)

**🧠 Model Optimization**
* Transfer learning (ImageNet weights)
* Fine-tuning (partial unfreezing of layers)
* Dropout regularization
* L2 weight decay

**📉 Training Strategies**
* Learning rate experimentation
* Baseline vs improved model comparison
* Scratch vs pretrained evaluation

## 📈 Evaluation Metrics

Models are evaluated using:

* Accuracy / Validation Accuracy
* Loss curves
* Confusion Matrix
* Classification Report (Precision, Recall, F1-score)
* ROC-AUC analysis
* Training time comparison
* Parameter efficiency (accuracy per model size)

## ⚙️ System-Level Analysis

The project includes hardware-aware profiling:

* CPU and RAM utilization
* GPU detection via TensorFlow + NVIDIA-SMI
* Memory optimization (clear_session, garbage collection)
* Training efficiency analysis across architectures

## 🏆 Key Findings
* Transfer learning consistently outperforms scratch training
* EfficientNetB0 and ResNet50 achieve the best balance of accuracy and efficiency
* MobileNetV2 is ideal for lightweight deployment
* Overfitting is significantly reduced using:
  * Dropout
  * L2 regularization
  * Fine-tuning strategies
* Learning rate selection has a critical impact on convergence stability

## 📁 Project Structure

```text
├── data_loading_and_analysis
├── dataset_distribution_analysis
├── augmentation_visualization
├── model_architectures
├── baseline_training_pipeline
├── improved_training_strategies
├── evaluation_and_metrics
├── performance_comparison
└── scenario_based_experiments 
```

## 📌 Technologies Used
* TensorFlow / Keras
* NumPy & Pandas
* Matplotlib & Seaborn
* Scikit-learn
* KaggleHub
* Psutil


## ⭐ Project Goal
This project aims to demonstrate how different deep learning architectures behave in a real-world medical imaging task and to provide a systematic comparison of modern CNN-based models.
