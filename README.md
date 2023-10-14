# MNIST Handwritten Digit Recognition with Multiple Machine Learning Models
In this project, a variety of machine learning models, including Support Vector Machines (SVM), k-Nearest Neighbors (k-NN), Logistic Regression, Decision Trees, and Random Forest Classifier, were applied to perform handwritten digit recognition on the MNIST dataset. The models were trained using the "mnist_train.csv" dataset, and a structured approach was taken with two key functions: one for training and another for testing. These trained models were saved in .pkl (pickle) format for future use.

Before training the models, the dataset was preprocessed using the Min-Max scaling method (MinMaxScaler) to normalize the data.

For model evaluation, various performance metrics such as accuracy, precision, recall, confusion matrices, F1 scores, and classification report were used to assess the models' effectiveness.

In a separate Jupyter Notebook file (Test_Five_ML_Pretrained_Model.ipynb), the saved machine learning models were loaded, and predictions were made on randomly selected handwritten digits from the "mnist_test.csv dataset". The project's findings and results were visualized using the Matplotlib and Seaborn libraries.

Overall, this project encompasses the training of multiple machine learning models for MNIST digit recognition, model evaluation with various metrics, and the application of these models to predict and visualize random handwritten digits from the test dataset.
# Dataset description
**1. Content**:
* The MNIST dataset contains a set of 28x28 pixel grayscale images of handwritten digits (0 to 9).
* Each image is labeled with the corresponding digit it represents.
  
**2. Size**:
* The dataset consists of 60,000 training images and 10,000 test images.
* It's often divided into a training set and a test set, making it suitable for training and evaluating machine learning models.

**3. Usage**:
* MNIST is frequently used for tasks like digit recognition, image classification, and deep learning model benchmarking.
* Researchers and machine learning practitioners often use MNIST as a starting point for experimenting with various machine learning algorithms.

  **Note  :** Only mnist_test.csv file uploaded, for mnist_train.csv dataset you can download from Kaggle as well as GitHub.
