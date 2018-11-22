import pandas as pd
import numpy as np
from numpy import linalg as LA
import time
import math

precision_k = 5000
num_of_users = 6040 + 1
num_of_movies= 3952 + 1
num_of_ratings = 1000209
def main():
    #Reading ratings file:
    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    ratings = pd.read_csv('ml-1m/ratings.dat', sep="::", names=r_cols,encoding='latin-1',engine='python')
    ratings = ratings[['user_id', 'movie_id', 'rating']]
    ratings_list = ratings.values.tolist()
    user_movie_matrix = np.zeros((num_of_users,num_of_movies))
    #making the utlity matrix
    for i in range(num_of_ratings):
        user_id = ratings_list[i][0]
        movie_id = ratings_list[i][1]
        rating = ratings_list[i][2]
        user_movie_matrix[user_id][movie_id] = rating
    
    matrix_centered_zero = np.copy(user_movie_matrix)
    mean = 0.0
    #centering the matrix about mean
    for i in range(1,num_of_users):
        sum = 0.0
        count = 0.0
        #computing mean
        for j in  range(1,num_of_movies):
            if(user_movie_matrix[i][j] != 0):
                sum = sum + user_movie_matrix[i][j]
                count = count + 1.0
        mean = sum / count 
        #centering data
        for j in range(1,num_of_movies):
            if(user_movie_matrix[i][j] == 0.0):
                matrix_centered_zero[i][j] = mean
            else:
                matrix_centered_zero[i][j] = matrix_centered_zero[i][j] - mean
    #making the training data set with the first 1000 * 1000 ratings as missing by assigning as -1
    test = np.copy(matrix_centered_zero)
    for i in range(1,1001):
        for j in range(1,1001):
            if(matrix_centered_zero[i][j] != 0):
                test[i][j] = -1
    
    mean = 0.0
    #centering the training data set
    for i in range(1,num_of_users):
        sum = 0.0
        count = 0.0
        #computing mean
        for j in  range(1,num_of_movies):
            if(test[i][j] == -1):
                sum = sum + 0.0
                count = count + 1.0
            elif(test[i][j] > 0):
                sum = sum + test[i][j]
                count = count + 1.0
        mean = sum / count 
        #centering data
        for j in range(1,num_of_movies):
            if(test[i][j] == -1 or test[i][j] == 0):
                test[i][j] = mean
            else:
                test[i][j] = test[i][j] - mean
    precision_rating = []
    start = time.time()
    #computing A(transpose) * A
    AtA = np.dot(np.transpose(test), test)
    #computing the eigen vales and vectors
    eigen_values_V , eigen_vectors_V = LA.eig(AtA)
    #retaining only the real part of eigen values and vectors
    eigen_values_V = np.real(eigen_values_V)
    eigen_vectors_V = np.real(eigen_vectors_V)
    #sorting the eigen values in descending order
    idV = np.argsort(np.multiply(-1,eigen_values_V))
    eigen_values_V = eigen_values_V[idV]
    #rearranging vectors as per the eigen values
    eigen_vectors_V = eigen_vectors_V[:, idV]
    #making the sigma matrix
    S = np.sqrt(np.abs(eigen_values_V))
    S = np.diag(S)
    #computing the inverse of sigma matrix
    Sinv = np.linalg.pinv(S)
    #computing U
    U = np.matmul(np.matmul(test, eigen_vectors_V), Sinv)
    #computing the original matrix from the SVD decomposition
    answer = np.matmul(np.matmul(U, S), np.transpose(eigen_vectors_V))
    squares_sum = 0.0
    count_sq = 0.0
    for i in range(1,1001):
        for j in range(1,1001):
            if(matrix_centered_zero[i][j] != 0):
                precision_rating.append(answer[i][j])
                print("Actual rating")
                print(matrix_centered_zero[i][j])
                print("Predicted rating")
                print(answer[i][j])
                #computing rmse
                squares_sum = squares_sum + (answer[i][j] - matrix_centered_zero[i][j])**2
                count_sq = count_sq + 1.0
                print("Root mean squared error")
                print(math.sqrt(squares_sum / count_sq))
                if(count_sq > 1):
                    print("Spearman's correlation")
                    correlation = 1 - ((6 * squares_sum) / (count_sq**3 - count_sq))
                    print(correlation)
                print("")
    #calculation of the precision at top k
    precision_rating.sort(reverse=True)
    countk = 0.0
    for i in range(0, precision_k):
        if(precision_rating[i] >= 3):
            countk = countk + 1
    precision_at_topk = countk / precision_k
    print("Precision at top k")
    print(precision_at_topk)


    
if __name__== "__main__":
    main()