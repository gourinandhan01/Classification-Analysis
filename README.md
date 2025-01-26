# Breast Cancer Classification Using Supervised Learning

## Objective
The objective of this project is to evaluate the performance of five classification algorithms on the breast cancer dataset and compare their results. The algorithms used are:
1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. Support Vector Machine (SVM)
5. k-Nearest Neighbors (k-NN)

---

## Dataset
The dataset used is the breast cancer dataset from the `sklearn.datasets` library. It contains 30 features that describe characteristics of cell nuclei present in breast cancer biopsies. The target variable is binary: 0 for malignant and 1 for benign tumors.

---

## Preprocessing
The dataset is preprocessed by:
1. **Scaling** the features using `StandardScaler` to ensure that all features have the same scale.
2. **Handling missing values** (although there were none in this dataset).

---

## Algorithms Implemented
The following classification algorithms are implemented:
1. **Logistic Regression**: A linear model for binary classification.
2. **Decision Tree Classifier**: A tree-based model that splits the data based on feature values.
3. **Random Forest Classifier**: An ensemble method using multiple decision trees.
4. **Support Vector Machine (SVM)**: A model that finds the optimal hyperplane to separate the classes.
5. **k-Nearest Neighbors (k-NN)**: A non-parametric model that classifies based on the majority class of the nearest neighbors.

---

## Performance Comparison
The models were evaluated based on accuracy. The following results were obtained:

| Model                | Accuracy   |
|----------------------|------------|
| Logistic Regression  | 97.37%     |
| Support Vector Machine (SVM) | 97.37%     |
| Random Forest        | 96.49%     |
| Decision Tree        | 94.74%     |
| k-Nearest Neighbors (k-NN) | 94.74%     |

### Best Performing Algorithm:
- **Logistic Regression** and **SVM** both performed the best with an accuracy of **97.37%**.

### Worst Performing Algorithm:
- **Decision Tree** and **k-NN** both had the lowest accuracy of **94.74%**. These models showed some overfitting or sensitivity to feature scaling.

---

## Installation and Usage

### Prerequisites:
To run this project, you will need to install the following Python libraries:
- `numpy`
- `pandas`
- `scikit-learn`

