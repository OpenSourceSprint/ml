import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

dataset = pd.read_csv("diabetes.csv")
# print(dataset.head())

# print(dataset.describe())

X = dataset.iloc[:,0:8]
y = dataset.iloc[:,8]

# Standardize feature space mean 0 and variance 1
X_std = (X-np.mean(X,axis = 0))/np.std(X,axis = 0)



# covariance matrix for X (8x8)
cov_matrix = np.cov(X_std, rowvar=False)
# print(f'Covariance matrix of X:\n {cov_matrix}')



# eigenvalues(8x1) and eigenvectors(8x8)
eigenvalues, eigenvectors, = np.linalg.eig(cov_matrix)

# print(f'Eigenvectors of Cov(X): \n {eigenvectors}')

# print(f'\nEigenvalues of Cov(X): \n {eigenvalues}')


# Set of (eigenvalue, eigenvector) pairs
eig_pairs = [(eigenvalues[index], eigenvectors[:,index]) for index in range(len(eigenvalues))]

# Descending sort (eigenvalue, eigenvector) pairs with respect to eigenvalue
eig_pairs.sort()
eig_pairs.reverse()

eigvalues_sort = [eig_pairs[index][0] for index in range(len(eigenvalues))]
eigvectors_sort = [eig_pairs[index][1] for index in range(len(eigenvalues))]

# print(f'Eigenvalues in descending order: \n{eigvalues_sort}')


# cumulative variance of each principle component
var_comp_sum = np.cumsum(eigvalues_sort)/sum(eigvalues_sort)

# cumulative proportion of varaince with respect to components
# print(f"Cumulative proportion of variance explained vector: \n {var_comp_sum}")

# x-axis for number of principal components kept
num_comp = range(1,len(eigvalues_sort)+1)

plt.title('Cum. Prop. Variance Explain and Components Kept')

# x-label
plt.xlabel('Principal Components')

# y-label
plt.ylabel('Cum. Prop. Variance Expalined')

# Scatter plot 
plt.scatter(num_comp, var_comp_sum)
plt.show()


# Project data onto 2d 

# Keep the first two principal components 
# P_reduce is 8 x 2 matrix
P_reduce = np.array(eigvectors_sort[0:2]).transpose()

# The projected data in 2D will be n x 2 matrix
Proj_data_2D = np.dot(X_std,P_reduce)



# Plot projected the data onto 2D (test negative for diabetes)
negative = plt.scatter(Proj_data_2D[:,0][y == 0], Proj_data_2D[:,1][y == 0])

# Plot projected the data onto 2D (test positive for diabetes)
positive = plt.scatter(Proj_data_2D[:,0][y == 1], Proj_data_2D[:,1][y == 1], color = "red")


plt.title('PCA Dimensionality Reduction to 2D')

# y-label
plt.ylabel('Principal Component 2')

# x-label
plt.xlabel('Principal Component 1')

# legend
plt.legend([negative,positive],["No Diabetes", "Have Diabetes"])

plt.show()


# data onto 3d 
# P_reduce is k x 3 matrix
P_reduce = np.array(eigvectors_sort[0:3]).transpose()

# Let's project data onto 3D space
# The projected data in 3D will be n x 3 matrix
Proj_data_3D = np.dot(X_std,P_reduce)

# Visualize data in 3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot in 3D (test negative for diabetes)
negative = ax.scatter(Proj_data_3D[:,0][y == 0], Proj_data_3D[:,1][y == 0], Proj_data_3D[:,2][y == 0], label="No Diabetes")

# Scatter plot in 3D (test positive for diabetes)
positive = ax.scatter(Proj_data_3D[:,0][y == 1], Proj_data_3D[:,1][y == 1], Proj_data_3D[:,2][y == 1], color="red", label="Have Diabetes")

ax.set_title('PCA Reduces Data to 3D')

# x-label 
ax.set_xlabel('Principal Component 1')

# y-label
ax.set_ylabel('Principal Component 2')

# z-label
ax.set_zlabel('Principal Component 3')

ax.legend()

plt.show()
