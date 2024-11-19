
# **🪟 Image Classification Using Neural Networks**

This project demonstrates image classification using **Artificial Neural Networks (ANN)** and **Convolutional Neural Networks (CNN)** with the **CIFAR-10 dataset**. It includes dataset preprocessing, model building, training, and evaluation.

---

## **✨ Table of Contents**

1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Technologies Used](#technologies-used)
4. [Project Structure](#project-structure)
5. [How to Run](#how-to-run)
6. [Code Explanation](#code-explanation)
7. [Results](#results)
8. [Conclusion](#conclusion)

---

## **📝 Introduction**
Image classification is the process of identifying and categorizing objects in an image. In this project, we use two types of neural networks:
- **ANN** (Artificial Neural Network): A basic model for image classification.
- **CNN** (Convolutional Neural Network): A more advanced model that captures image-specific features for better accuracy.

---

## **💾 Dataset**
We use the **CIFAR-10 dataset**, which contains:
- 60,000 images divided into 10 categories: *airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck*.
- Images are 32x32 pixels with 3 color channels (RGB).

---

## **💻 Technologies Used**
- **Python**: Programming language.
- **TensorFlow/Keras**: Framework for building and training neural networks.
- **Matplotlib**: Library for data visualization.
- **NumPy**: Library for numerical computations.

---

## **🗂️ Project Structure**
```
.
├── README.md       # Documentation
└── Image_Classification_(Cifar_10).ipynb         # Main script to execute the project
```

---

## **⚡ How to Run**

1. **Clone the repository**:
   ```bash
   git clone https://github.com/DevSumanP/ImageClassifier101.git
   cd ImageClassifier101
   ```

2. **Install dependencies**:
   ```bash
   pip install tensorflow matplotlib numpy
   ```

3. **Run the script**:
   ```bash
   python main.py
   ```

4. **Expected Output**:
   - Model training logs.
   - Accuracy metrics for both ANN and CNN.
   - Visualizations of predictions.

---

## **⚡ Code Explanation**

1. **Data Preprocessing**:
   - Normalize pixel values to range [0, 1] for faster and stable training.
   - Reshape labels to make them compatible with the models.

2. **ANN Model**:
   - A simple architecture with fully connected layers.
   - Output layer uses a `softmax` activation function to predict probabilities for 10 classes.

3. **CNN Model**:
   - Includes convolutional and pooling layers to extract features from images.
   - Improves classification accuracy compared to ANN.

4. **Evaluation**:
   - Evaluate models using test data and generate classification reports.

---

## **😃 Results**

| **Model**         | **Accuracy** |
|--------------------|--------------|
| **ANN**           | ~49%         |
| **CNN**           | ~70%         |

---

## **Conclusion**

- **ANN** is suitable for simple datasets but lacks accuracy for images.
- **CNN** performs better due to its ability to extract spatial hierarchies (e.g., edges, textures).
- Image classification is an essential task in AI, with applications in self-driving cars, medical diagnostics, and more.

---

## **Future Work**
- Use data augmentation to improve accuracy.
- Experiment with deeper networks or transfer learning using pre-trained models.
- Apply this framework to a custom dataset for real-world applications.

---

## **🗒️ License**
This project is open-source under the [MIT License](LICENSE).
