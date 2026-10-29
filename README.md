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
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. Load the dataset
df = pd.read_csv("Iris.csv")

# Show the first five rows and the class counts
print("First five rows:")
print(df.head())

print("\nSpecies counts:")
print(df["Species"].value_counts())

# 2. Plot all three species using three measurements
colors = {
    "Iris-setosa": "red",
    "Iris-versicolor": "blue",
    "Iris-virginica": "green"
}

fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot(111, projection="3d")

for species, color in colors.items():
    rows = df[df["Species"] == species]
    ax.scatter(
        rows["SepalLengthCm"],
        rows["SepalWidthCm"],
        rows["PetalLengthCm"],
        color=color,
        label=species
    )

ax.set_title("Iris dataset: three species")
ax.set_xlabel("Sepal length (cm)")
ax.set_ylabel("Sepal width (cm)")
ax.set_zlabel("Petal length (cm)")
ax.legend()
plt.show()

# 3. Keep two species for binary classification
chosen_species = ["Iris-setosa", "Iris-versicolor"]
data = df[df["Species"].isin(chosen_species)].copy()

# 4. Select petal length and petal width
feature_names = ["PetalLengthCm", "PetalWidthCm"]
X = data[feature_names].to_numpy(dtype=float)

# Encode the species as -1 and +1
label_map = {
    "Iris-setosa": -1,
    "Iris-versicolor": 1
}
y = data["Species"].map(label_map).to_numpy()

# Plot the two selected species using the original measurements
plt.figure(figsize=(7, 5))

for species, color in [
    ("Iris-setosa", "red"),
    ("Iris-versicolor", "blue")
]:
    rows = data[data["Species"] == species]
    plt.scatter(
        rows["PetalLengthCm"],
        rows["PetalWidthCm"],
        color=color,
        label=species
    )

plt.title("Petal measurements: setosa vs versicolor")
plt.xlabel("Petal length (cm)")
plt.ylabel("Petal width (cm)")
plt.legend()
plt.grid(True)
plt.show()

# 5. Split into training and test data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 6. Standardize the features using training data only
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 7. Train the perceptron
learning_rate = 0.1
epochs = 50

weights = [0.0, 0.0]
bias = 0.0
errors_per_epoch = []

for epoch in range(epochs):
    errors = 0

    for xi, target in zip(X_train, y_train):
        score = weights[0] * xi[0] + weights[1] * xi[1] + bias

        # Update weights and bias when the example is misclassified
        if target * score <= 0:
            weights[0] += learning_rate * target * xi[0]
            weights[1] += learning_rate * target * xi[1]
            bias += learning_rate * target
            errors += 1

    errors_per_epoch.append(errors)

    # Stop when there are no training mistakes
    if errors == 0:
        break

# 8. Test the perceptron on the held-out data
test_scores = X_test @ weights + bias
y_pred = [1 if score >= 0 else -1 for score in test_scores]

accuracy = accuracy_score(y_test, y_pred)

print("\nSelected records:", len(data))
print("Training records:", len(X_train))
print("Test records:", len(X_test))
print("Training epochs completed:", len(errors_per_epoch))
print("Test accuracy: {:.2f}%".format(accuracy * 100))
print("Confusion matrix (labels -1, +1):")
print(confusion_matrix(y_test, y_pred, labels=[-1, 1]))

# 9. Plot the two classes and the decision boundary
plt.figure(figsize=(7, 5))

for label, name, color in [
    (-1, "Iris-setosa", "blue"),
    (1, "Iris-versicolor", "orange")
]:
    mask = y_train == label
    plt.scatter(
        X_train[mask, 0],
        X_train[mask, 1],
        label=name,
        color=color
    )

# Draw the decision boundary: w1*x1 + w2*x2 + bias = 0
x_values = [
    X_train[:, 0].min() - 0.5,
    X_train[:, 0].max() + 0.5
]

if abs(weights[1]) > 1e-10:
    y_values = [
        -(weights[0] * x + bias) / weights[1]
        for x in x_values
    ]
    plt.plot(x_values, y_values, "k--", label="Decision boundary")

plt.xlabel("Standardized petal length")
plt.ylabel("Standardized petal width")
plt.title("Perceptron: Iris-setosa vs Iris-versicolor")
plt.legend()
plt.grid(True)
plt.show()

# 10. Plot training mistakes for each epoch
plt.figure(figsize=(7, 4))
plt.plot(
    range(1, len(errors_per_epoch) + 1),
    errors_per_epoch,
    marker="o"
)
plt.xlabel("Epoch")
plt.ylabel("Number of training mistakes")
plt.title("Perceptron training error by epoch")
plt.grid(True)
plt.show()

# OUTPUT:

<img width="810" height="296" alt="Screenshot 2026-10-29 202124" src="https://github.com/user-attachments/assets/d414065b-3483-4ee4-9245-caa689da43f4" />

<img width="773" height="1007" alt="Screenshot 2026-10-29 202243" src="https://github.com/user-attachments/assets/2e2797dc-c133-4cce-9346-8b2ac44e4677" />

<img width="855" height="721" alt="Screenshot 2026-10-29 202253" src="https://github.com/user-attachments/assets/5c43c749-31a4-47e3-be2b-6c9434d506bb" />

 <img width="792" height="447" alt="Screenshot 2026-10-29 202313" src="https://github.com/user-attachments/assets/51e8c1a6-b79c-411a-aca0-c9aac97ee42e" />


# RESULT:
 Thus, a single layer perceptron model is implemented using python to classify Iris data set.

 
