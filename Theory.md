# Gaussian Naive Bayes Classifier for Breast Cancer Detection

## Introduction

This project implements a **Gaussian Naive Bayes** classifier to detect breast cancer based on the **Breast Cancer Wisconsin dataset**. The classifier is built using both **scikit-learn's GaussianNB model** and a custom Gaussian Naive Bayes implementation.

The Naive Bayes algorithm is a **probabilistic classifier** based on Bayes' Theorem, assuming that the features are conditionally independent given the class (i.e., the naive assumption). Gaussian Naive Bayes is a special case of Naive Bayes, which assumes that the continuous features follow a **Gaussian (normal) distribution**.

---

## Dataset

The dataset used in this project is the **Breast Cancer Wisconsin dataset**, which is available in `scikit-learn`. It contains 569 instances and 30 features, where each instance is a patient's breast cancer diagnostic measurements. The target variable is binary, representing:
- 0: Malignant (cancerous)
- 1: Benign (non-cancerous)

### Features:
- 30 real-valued input features, such as mean radius, mean texture, etc., calculated from a digitized image of a fine needle aspirate (FNA) of a breast mass.

---

## Theory: Gaussian Naive Bayes

Naive Bayes is based on **Bayes' Theorem**, which calculates the posterior probability of a class `C` given a set of features `X = {x1, x2, ..., xn}`.

### Bayes' Theorem:
$$
P(C | X) = \frac{P(X | C) \cdot P(C)}{P(X)}
$$

Where:
- \( P(C | X) \): Posterior probability of class `C` given the feature vector `X`
- \( P(X | C) \): Likelihood of feature vector `X` given class `C`
- \( P(C) \): Prior probability of class `C`
- \( P(X) \): Evidence (total probability of `X`)

In **Gaussian Naive Bayes**, we assume that the likelihood \( P(X | C) \) is a Gaussian (normal) distribution, parameterized by the mean and variance of each feature in the training data for each class.

### Gaussian Probability Density Function (PDF):
For a given feature `x` with mean `μ` and variance `σ²`, the probability density function is given by:

$$
P(x | C) = \frac{1}{\sqrt{2\pi\sigma^2}} \cdot \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)
$$

The **Gaussian Naive Bayes algorithm** combines the likelihoods of each feature for both classes and multiplies them with the class prior to predict the class with the highest posterior probability.

---

## Project Workflow

### 1. Data Preprocessing
- Load the Breast Cancer dataset using `scikit-learn`'s `load_breast_cancer()` function.
- Create a Pandas DataFrame containing the features and the target.
- Split the data into training (80%) and testing (20%) sets using `train_test_split`.


### 2. Custom Gaussian Naive Bayes Implementation
The custom implementation of Gaussian Naive Bayes performs the following steps:
1. **Calculate mean and variance** for each feature within each class on the training data.
2. **Calculate prior probabilities** based on the frequency of each class in the training set.
3. **Predict class** for new data by computing the posterior probability using Bayes' Theorem and Gaussian PDF.
4. Evaluate the accuracy of the custom implementation by comparing predictions to true labels.

### 3. Model Evaluation
- **Accuracy** is calculated as the proportion of correct predictions out of the total number of predictions.

### 4. scikit-learn Gaussian Naive Bayes
- Train the model using `GaussianNB()` from `scikit-learn`.
- Evaluate the model's accuracy on the test data.
