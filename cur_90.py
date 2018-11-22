import pandas as pd
import numpy as np
from numpy import linalg as LA
import math
import random
import time
num_of_users = 6040 + 1
num_of_movies= 3952 + 1
num_of_ratings = 1000209
precision_k = 5000
def main():
    #Reading ratings file:
    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    ratings = pd.read_csv('ml-1m/ratings.dat', sep="::", names=r_cols,encoding='latin-1',engine='python')

    ratings = ratings[['user_id', 'movie_id', 'rating']]
    ratings_list = ratings.values.tolist()
    user_movie_matrix = np.zeros((num_of_users,num_of_movies))
    #computing the utilty matrix
    for i in range(num_of_ratings):
        user_id = ratings_list[i][0]
        movie_id = ratings_list[i][1]
        rating = ratings_list[i][2]
        user_movie_matrix[user_id][movie_id] = rating

    matrix_centered_zero = np.copy(user_movie_matrix)
    #centering the test data set
    mean = 0.0
    for i in range(1,num_of_users):
        sum = 0.0
        count = 0.0
        #calculating mean
        for j in  range(1,num_of_movies):
            if(user_movie_matrix[i][j] != 0):
                sum = sum + user_movie_matrix[i][j]
                count = count + 1.0
        mean = sum / count 
        #centering the data about mean
        for j in range(1,num_of_movies):
            if(user_movie_matrix[i][j] == 0.0):
                matrix_centered_zero[i][j] = mean
            else:
                matrix_centered_zero[i][j] = matrix_centered_zero[i][j] - mean
    
    test = np.copy(matrix_centered_zero)
    #making the training data set with the first 1000 * 1000 values as -1
    for i in range(1,1001):
        for j in range(1,1001):
            if(matrix_centered_zero[i][j] != 0):
                test[i][j] = -1
    #center the training data set
    mean = 0.0
    for i in range(1,num_of_users):
        sum = 0.0
        count = 0.0
        #calculating mean
        for j in  range(1,num_of_movies):
            if(test[i][j] == -1):
                sum = sum + 0.0
                count = count + 1.0
            elif(test[i][j] > 0):
                sum = sum + test[i][j]
                count = count + 1.0
        mean = sum / count 
        #centering the data about mean
        for j in range(1,num_of_movies):
            if(test[i][j] == -1 or test[i][j] == 0):
                test[i][j] = mean
            else:
                test[i][j] = test[i][j] - mean
  
    #k factor for CUR   
    k = 250
    #computing the sum of all elements squared
    total_sum_sq = 0.0
    for i in range(1,num_of_users):
        for j in range(1,num_of_movies):
            total_sum_sq = total_sum_sq + (test[i][j])**2

    #computing the probability distribution for all the columns
    col_dis_pr = []
    col_dis_pr.append(0.0)
    for i in range(1,num_of_movies):
        col_sum_sq = 0.0
        for j in range(1,num_of_users):
            col_sum_sq = col_sum_sq + (test[j][i])**2
        col_dis_pr.append(col_sum_sq / total_sum_sq)

    #computing the probabilty distribution for all the rows
    row_dis_pr = []
    row_dis_pr.append(0.0)
    for i in range(1,num_of_users):
        row_sum_sq = 0.0
        for j in range(1,num_of_movies):
            row_sum_sq = row_sum_sq + (test[i][j])**2
        row_dis_pr.append(row_sum_sq / total_sum_sq)

    #computing a list with indices of all columns
    cols_index = []
    cols_index.append(-1)
    for i in range(0,3952):
        cols_index.append(i+1)

    #computing a list with indices of all rows
    rows_index = []
    rows_index.append(-1)
    for i in range(0,6040):
        rows_index.append(i+1)

    #computing random values with given probability distribution
    cols = np.random.choice(cols_index, 4 * k,replace=False, p = col_dis_pr)
    rows = np.random.choice(rows_index, 4 * k,replace=False, p = row_dis_pr)

    #c  = 4 * k
    c_attr = 1000.0
    C = np.zeros((num_of_users, 4*k + 1))
    #Computing C
    for i in range(1,4*k+1):
        C[:,i] = np.divide(test[:,cols[i-1]], np.sqrt(np.multiply(c_attr,col_dis_pr[cols[i-1]])))

    R = np.zeros((4*k+1, num_of_movies))
    #Computing R
    for i in range(1,4*k+1):
        R[i,:] = np.divide(test[rows[i-1],:], np.sqrt(np.multiply(c_attr,row_dis_pr[rows[i-1]])))

    #Computing their pseudoinverses
    C_inv = np.linalg.pinv(C)
    R_inv = np.linalg.pinv(R)
    #Computing U
    U = np.matmul(np.matmul(C_inv, test), R_inv)

    #Computing SVD of U
    AtA = np.dot(np.transpose(U), U)
    #Computing eigen values and vectors
    eigen_values_V , eigen_vectors_V = LA.eig(AtA)
    #discarding the imaginary part of eigen vectors and values
    eigen_values_V = np.real(eigen_values_V)
    eigen_vectors_V = np.real(eigen_vectors_V)
    #sorting the eigen values in descending order
    idV = np.argsort(np.multiply(-1,eigen_values_V))
    eigen_values_V = eigen_values_V[idV]
    #rearranging eigen vectors with respect to eigen values
    eigen_vectors_V = eigen_vectors_V[:, idV]
    #computing the sigma matrix
    S = np.sqrt(np.abs(eigen_values_V))
    S = np.diag(S)
    #computing the inverse of the sigma matrix
    Sinv = np.linalg.pinv(S)
    #computing the left SVD matrix
    W = np.matmul(np.matmul(U, eigen_vectors_V), Sinv)
    #Computing the 90% sigma matrix
    energy = 0.0
    #computing the total energy
    for i in range(S.shape[0]):
        energy = energy + S[i][i]**2
    #computing 90% of total energy
    ninety_percent = 0.9 * energy
    cut = 0
    while(energy > ninety_percent):
        temp_energy = 0.0
        #compute leaving the last 'cut' elements
        for i in range(S.shape[0] - cut):
            temp_energy = temp_energy + S[i][i] ** 2 
        #check if its more than 90%
        if(temp_energy > ninety_percent):
            #if yes then update cut
            cut = cut + 1
            continue
        else:
            break
    size_S = S.shape[0]
    new_shape = size_S - cut + 1
    #reshape all the matrices
    new_S = S[0:new_shape,0:new_shape]
    new_U = W[:,0:new_shape]
    R_new = eigen_vectors_V[0:new_shape, :]
    #Compute the U in CUR  
    answer = np.matmul(np.matmul(new_U, new_S), (R_new))
    #Compute the original matrix
    answer = np.matmul(np.matmul(C, answer), R)

    squares_sum = 0.0
    count_sq = 0.0
    precision_rating = []
    for i in range(1,1001):
        for j in range(1,1001):
            if(user_movie_matrix[i][j] != 0):
                precision_rating.append(answer[i][j])
                print("Actual rating")
                print(matrix_centered_zero[i][j])
                print("Predicted rating")
                print(answer[i][j])
                #compute rmse
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
        if(precision_rating[i] >= 1.0):
            countk = countk + 1
    precision_at_topk = countk / precision_k
    print("Precision at top k")
    print(precision_at_topk)

if __name__ == "__main__":
    main()

