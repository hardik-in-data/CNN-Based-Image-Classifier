# CNN-Based Image Classifier

## Overview
This project implements a **Convolutional Neural Network (CNN)** for image classification using **TensorFlow** and **Keras**. The model is trained on the **CIFAR-10 dataset**, demonstrating its capability to recognize and categorize different object classes. It covers the entire deep learning workflow, including data preprocessing, model design, training, evaluation, and real-world testing.

## Key Features
- **Deep Learning Architecture**: Implements a CNN designed for image recognition, leveraging convolutional, pooling, and dense layers.
- **Optimized Performance**: Incorporates hyperparameter tuning and data augmentation to enhance model generalisation.
- **Efficient Training**: Designed to run efficiently on systems without dedicated GPUs while ensuring high classification accuracy.
- **Evaluation & Validation**: Includes confusion matrix analysis, test set performance evaluation, and misclassification analysis.
- **Robust Predictions**: Supports batch processing of unseen images for real-world applications.

## Project Workflow
1. **Data Preprocessing**
   - Downloading and normalising the CIFAR-10 dataset.
   - Applying data augmentation techniques to improve model robustness.
   
2. **Model Architecture**
   - Implementing a CNN with multiple **Conv2D** layers.
   - Utilising **Batch Normalisation** and **Dropout** for improved performance.
   - Selecting optimal activation functions and an **Adam optimiser**.

3. **Training & Optimisation**
   - Hyperparameter tuning for enhanced classification performance.
   - Monitoring loss and accuracy trends using visualisation techniques.
   
4. **Evaluation**
   - Generating a **confusion matrix** to assess model performance.
   - Visualising misclassified images for further analysis.
   
5. **Real-World Testing**
   - Making predictions on external images.
   - Batch processing images without predefined class labels.

## 🛠 Installation
To set up the environment, install the required dependencies:

```bash
pip install tensorflow tensorflow-datasets numpy matplotlib
```

## 📂 Usage
### Running the Model
1. Clone the repository:
   ```bash
   git clone https://github.com/thenotorioushardik/CNN-Based-Image-Classifier.git
   cd CNN-Based-Image-Classifier
   ```
2. Open the Jupyter Notebook:
   ```bash
   jupyter notebook cnn_based_image_classifier.ipynb
   ```
3. Execute the notebook step by step to:
   - Load and preprocess the dataset.
   - Train the CNN model.
   - Evaluate its performance using test images.
   - Perform real-world classification on custom images.

## Model Performance Analysis
- Achieves **~80-90% accuracy** on CIFAR-10 test data.
- Regularisation techniques and data augmentation help mitigate overfitting.
- Real-world testing ensures adaptability to unseen data.

## 🤝 Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch (`feature-improvement`).
3. Commit your changes and submit a pull request.
