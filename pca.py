import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

dataset = pd.read_csv("data/diabetes.csv")
X = dataset.iloc[:,0:8]
y = dataset.iloc[:,8]
X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

cov_matrix = np.cov(X_std, rowvar=False)
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
eig_pairs = [(eigenvalues[index], eigenvectors[:,index]) for index in range(len(eigenvalues))]

eig_pairs.sort(reverse=True)
eigvalues_sort = [eig_pairs[index][0] for index in range(len(eigenvalues))]
eigvectors_sort = [eig_pairs[index][1] for index in range(len(eigenvalues))]

var_comp_sum = np.cumsum(eigvalues_sort) / sum(eigvalues_sort)

num_comp = range(1, len(eigvalues_sort) + 1)

plt.title('Cum. Prop. Variance Explain and Components Kept')
plt.xlabel('Principal Components')
plt.ylabel('Cum. Prop. Variance Expalined')
plt.scatter(num_comp, var_comp_sum)
plt.show()

P_reduce = np.array(eigvectors_sort[0:2]).T
Proj_data_2D = np.dot(X_std, P_reduce)

negative = plt.scatter(Proj_data_2D[:,0][y == 0], Proj_data_2D[:,1][y == 0])
positive = plt.scatter(Proj_data_2D[:,0][y == 1], Proj_data_2D[:,1][y == 1], color="red")

plt.title('PCA Dimensionality Reduction to 2D')
plt.ylabel('Principal Component 2')
plt.xlabel('Principal Component 1')
plt.legend([negative, positive], ["No Diabetes", "Have Diabetes"])
plt.show()

P_reduce = np.array(eigvectors_sort[0:3]).T
Proj_data_3D = np.dot(X_std, P_reduce)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
negative = ax.scatter(Proj_data_3D[:,0][y == 0], Proj_data_3D[:,1][y == 0], Proj_data_3D[:,2][y == 0], label="No Diabetes")
positive = ax.scatter(Proj_data_3D[:,0][y == 1], Proj_data_3D[:,1][y == 1], Proj_data_3D[:,2][y == 1], color="red", label="Have Diabetes")

ax.set_title('PCA Reduces Data to 3D')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.legend()
plt.show()
