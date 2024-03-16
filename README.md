# Breast-Cancer--Classifier
Breast Cancer Diagnosis with Decision Tree Classifier

Breast cancer is the most frequent cause of cancer mortality among women, emphasizing the importance of early detection to decrease mortality rates. In this project, we aim to work with the Breast Cancer Wisconsin (Diagnostic) Data Set, which can be downloaded from the UCI Machine Learning Repository.

## Dataset Overview
The dataset consists of 569 data points, with 212 labeled as Malignant and 357 labeled as Benign. These data points are derived from digitized images of fine needle aspirates (FNA) of breast masses, and the features describe characteristics of cell nuclei present in the images.

## Project Goals

### Data Loading and Splitting
- Utilize NumPy or Pandas package to load the dataset from "breast_cancer_wisconsin.csv".
- Split the dataset into training and testing sets with a test ratio of 0.3.

### Decision Tree Classification
- Define a Decision Tree classifier using the scikit-learn package with custom hyperparameters.
- Fit the classifier to the training set.
- Evaluate the classifier's precision, recall, F-score, and accuracy on both the training and testing sets.
- Plot confusion matrices of the model on both training and testing sets.

### Hyperparameter Study
- Investigate how different maximum tree depths and cost functions influence the efficiency of the Decision Tree on the provided dataset.
- Describe findings regarding:
  - Three different cost functions: ['gini', 'entropy', 'log_loss']
  - Six different maximum tree depths: [2, 4, 6, 8, 10, 12]

### Decision Boundary Visualization
- Depict a plot of the decision boundary based on the two mentioned hyperparameters.
- Provide concise commentary on fundamental features observed.

## Usage
1. Clone the repository to your local machine.
2. Download the dataset from the provided link and place it in the appropriate directory.
3. Execute the provided Python script to run the analysis and visualize the results.

## Contributors
- Prabhneet Singh
- prabhneets50@gmail.com

Feel free to contribute by forking the repository and submitting pull requests with improvements or additional analyses.

## Acknowledgments
- Dataset Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
- Special thanks to the scikit-learn and NumPy/Pandas development teams for their invaluable contributions to the machine learning community.
