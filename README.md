<H3>Name: Naveen R </H3>
<H3>Register no:212225040276 </H3>
<H3>Experiment No. 2 </H3>
## Implementation of Perceptron for Binary Classification

# AIM:
To implement a perceptron for classification using Python<BR>

# EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

# RELATED THEORETICAL CONCEPT:
A Perceptron is a basic learning algorithm invented in 1959 by Frank Rosenblatt. It is meant to mimic the working logic of a biological neuron. The human brain is basically a collection of many interconnected neurons. Each one receives a set of inputs, applies some sort of computation on them and propagates the result to other neurons.<BR>
A Perceptron is an algorithm used for supervised learning of binary classifiers.Given a sample, the neuron classifies it by assigning a weight to its features. To accomplish this a Perceptron undergoes two phases: training and testing. During training phase weights are initialized to an arbitrary value. Perceptron is then asked to evaluate a sample and compare its decision with the actual class of the sample.If the algorithm chose the wrong class weights are adjusted to better match that particular sample. This process is repeated over and over to finely optimize the biases. After that, the algorithm is ready to be tested against a new set of completely unknown samples to evaluate if the trained model is general enough to cope with real-world samples.<BR>
The important Key points to be focused to implement a perceptron:
Models have to be trained with a high number of already classified samples. It is difficult to know a priori this number: a few dozen may be enough in very simple cases while in others thousands or more are needed.
Data is almost never perfect: a preprocessing phase has to take care of missing features, uncorrelated data and, as we are going to see soon, scaling.<BR>
Perceptron requires linearly separable samples to achieve convergence.
The math of Perceptron. <BR>
If we represent samples as vectors of size n, where ‘n’ is the number of its features, a Perceptron can be modeled through the composition of two functions. The first one f(x) maps the input features  ‘x’  vector to a scalar value, shifted by a bias ‘b’
f(x)=w.x+b
 <BR>
A threshold function, usually Heaviside or sign functions, maps the scalar value to a binary output:

 


<img width="283" alt="image" src="https://github.com/Lavanyajoyce/Ex-2--NN/assets/112920679/c6d2bd42-3ec1-42c1-8662-899fa450f483">


Indeed if the neuron output is exactly zero it cannot be assumed that the sample belongs to the first sample since it lies on the boundary between the two classes. Nonetheless for the sake of simplicity,ignore this situation.<BR>


# ALGORITHM:
STEP 1: Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Plot the data to verify the linear separable dataset and consider only two classes<BR>
STEP 4:Convert the data set to scale the data to uniform range by using Feature scaling<BR>
STEP 4:Split the dataset for training and testing<BR>
STEP 5:Define the input vector ‘X’ from the training dataset<BR>
STEP 6:Define the desired output vector ‘Y’ scaled to +1 or -1 for two classes C1 and C2<BR>
STEP 7:Assign Initial Weight vector ‘W’ as 0 as the dimension of ‘X’
STEP 8:Assign the learning rate<BR>
STEP 9:For ‘N ‘ iterations ,do the following:<BR>
        v(i) = w(i)*x(i)<BR>
         
        W (i+i)= W(i) + learning_rate*(y(i)-t(i))*x(i)<BR>
STEP 10:Plot the error for each iteration <BR>
STEP 11:Print the accuracy<BR>
# PROGRAM:
# EXPERIMENT 2: PERCEPTRON FOR BINARY CLASSIFICATION

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 2. Load Dataset
df = pd.read_csv('/content/classification.csv')

print("Dataset Loaded Successfully")
print("\nFirst 5 Rows:")
print(df.head())
print("\nDataset Shape:", df.shape)

# 3. Explore Dataset
print("\nMissing Values:")
print(df.isnull().sum())

print("\nClass Distribution:")
print(df['y'].value_counts())

# 4. Visualize Class Distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='y')
plt.title("Binary Class Distribution")
plt.xlabel("Target Class")
plt.ylabel("Number of Samples")
plt.show()

# 5. Select Input and Target
X = df[['X2']]
y = df['y']

# 6. Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# 7. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 8. Train Perceptron Model
model = Perceptron(
    max_iter=1000,
    eta0=1.0,
    tol=0.001,
    random_state=42
)

model.fit(X_train_scaled, y_train)

print("\nPerceptron Model Trained Successfully")
print("Number of Iterations:", model.n_iter_)

# 9. Make Predictions
y_train_pred = model.predict(X_train_scaled)
y_pred = model.predict(X_test_scaled)

print("\nActual Values:")
print(y_test.values[:15])

print("\nPredicted Values:")
print(y_pred[:15])

# 10. Calculate Accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_pred)

print("\nTraining Accuracy:", round(train_accuracy * 100, 2), "%")
print("Testing Accuracy:", round(test_accuracy * 100, 2), "%")

# 11. Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 12. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=['Class 0', 'Class 1'],
    yticklabels=['Class 0', 'Class 1']
)
plt.title("Perceptron Confusion Matrix")
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
plt.show()

# 13. Error Analysis Across Epochs
errors = []

for epoch in range(1, 51):
    temp_model = Perceptron(
        max_iter=epoch,
        eta0=1.0,
        tol=None,
        shuffle=False,
        random_state=42
    )
    temp_model.fit(X_train_scaled, y_train)
    prediction = temp_model.predict(X_train_scaled)
    errors.append(np.sum(prediction != y_train))

plt.figure(figsize=(8, 5))
plt.plot(range(1, 51), errors, marker='o')
plt.title("Perceptron Errors Across Epochs")
plt.xlabel("Epoch")
plt.ylabel("Number of Errors")
plt.grid(True)
plt.show()


# OUTPUT:
<img width="1035" height="786" alt="Screenshot 2026-10-30 142331" src="https://github.com/user-attachments/assets/746d748d-d2f4-412d-b881-6987e3d64ae3" />

<img width="627" height="512" alt="Screenshot 2026-10-30 142436" src="https://github.com/user-attachments/assets/321fa61e-db7a-4455-960c-f0ba940f8c5b" />

<img width="875" height="1078" alt="Screenshot 2026-10-30 142458" src="https://github.com/user-attachments/assets/810546d9-c71e-4e1a-85fe-2aa6c679a83c" />

<img width="442" height="202" alt="Screenshot 2026-10-30 142512" src="https://github.com/user-attachments/assets/13799717-7c68-4094-9a27-37c45313e1df" />


# RESULT:
 Thus, a single layer perceptron model is implemented using python to classify Iris data set.

 
