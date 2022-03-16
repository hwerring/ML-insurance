import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

df = pd.read_csv("insurance.csv") # Reads the dataset to the variable df

# Assigns numerical values for discrete variables
smoker_dict = {
    'yes' : 1,
    'no' : 0
}
sex_dict = {
    'male' : 0,
    'female' : 1
}
region_dict = {
    'northeast' : 1,
    'southeast' : 2,
    'southwest' : 3,
    'northwest' : 4
}

# Puts columns in the dataset into vectors and converts discrete variables to numerical values
X_bmi = np.asarray(df.bmi.values)
X_children = np.asarray(df.children.values)
X_age = np.asarray(df.age.values)

X_smoking = np.asarray(df.smoker.values)
X_smoking = np.asarray([smoker_dict[i] for i in X_smoking])
X_sex = np.asarray(df.sex.values)
X_sex = np.asarray([sex_dict[i] for i in X_sex])
X_region = np.asarray(df.region.values)
X_region = np.asarray([region_dict[i] for i in X_region])

# Makes column y of outputs
y = np.asarray(df.charges.values)

def feature_scaling_normalisation(X):
    """Normalises a vector to range [-1, 1]"""
    X_norm = X

    mu = X.mean()
    sigma = X.std(ddof=1)
    N = X.shape[0]

    mu_matrix = np.multiply(np.ones(N), mu).T
    sigma_matrix = np.multiply(np.ones(N), sigma).T

    X_norm = np.subtract(X, mu).T
    X_norm = X_norm / sigma.T

    return [X_norm, mu, sigma]

def normalise(n, mean, sd):
    return (n - mean)/sd

def unnormalise(n, mean, sd):
    return n * sd + mean

def cost_LMS(X, y, Theta):
    """Implementation of the least squares cost function"""
    cost = 0
    for i in range(len(X)):
        cost += (np.dot(X[i], Theta) - y[i]) ** 2
    return cost/(2 * len(X))

# Normalises our training sets variables
bmi_norm = np.asarray(feature_scaling_normalisation(X_bmi)[0]).T
children_norm = np.asarray(feature_scaling_normalisation(X_children)[0]).T
smoking_norm = np.asarray(feature_scaling_normalisation(X_smoking)[0]).T
sex_norm = np.asarray(feature_scaling_normalisation(X_sex)[0]).T
age_norm = np.asarray(feature_scaling_normalisation(X_age)[0]).T
region_norm = np.asarray(feature_scaling_normalisation(X_region)[0]).T

# Creates a matrix where each column is the values of a variable
X_new = np.vstack((np.ones(len(y)), bmi_norm.T))
X_new = np.vstack((X_new, children_norm.T)).T
X_new = np.vstack((X_new.T, smoking_norm)).T
X_new = np.vstack((X_new.T, sex_norm)).T
X_new = np.vstack((X_new.T, age_norm)).T
X_new = np.vstack((X_new.T, region_norm)).T


def gradient_descent_LMS(X, y, theta, alpha):
    """Implementation of batch gradient descent using linear algebra"""
    return theta + alpha/len(y) * np.sum(np.multiply(y - np.sum(np.multiply(X, theta), axis = 1), X.T), axis = 1)
    
def plotting(X, y, theta):
    """Plots variable X against y with line of best fit described by theta"""
    plt.scatter(X[:,[1]], y, color='black')
    plt.plot(X[:,[1]], np.sum(np.multiply(X, theta), axis = 1), color='red', linewidth=1)

    plt.xlabel("BMI Normalised")
    plt.ylabel("Insurance")

    plt.show()

theta =  [0,0,0,0,0,0,0] # Assigns theta0 as a vector of zeroes
alpha1 = 0.1 # Sets learning rate to 0.1
for i in range(100):
    theta = gradient_descent_LMS(X_new, y, theta, alpha1) # Applies gradient descent algorithm 100 times
    if i % 10 == 0 or i < 11:
        print("After " + str(i) + " iterations, theta = " + str(np.round(theta,2)))
print("After 100 iterations, theta = " + str(np.round(theta,2)))

means = [feature_scaling_normalisation(i)[1] for i in [X_bmi, X_children, X_age, X_smoking, X_sex, X_region]]
sds = [feature_scaling_normalisation(i)[2] for i in [X_bmi, X_children, X_age, X_smoking, X_sex, X_region]]

def h(x, theta):
    """Converts discrete values to numerical values and normalises our input x, then outputs x.theta"""
    x[1] = sex_dict[x[1]]
    x[4] = smoker_dict[x[4]]
    x[5] = region_dict[x[5]]
    x = np.asarray([normalise(i, means[x.index(i)], sds[x.index(i)]) for i in x])
    x = np.insert(x, 0, 1)
    y = np.sum(np.multiply(x, theta))
    
    return y

new_input = [65, 'male', 26, 3, 'no', 'southeast']

print(h(new_input, theta))

