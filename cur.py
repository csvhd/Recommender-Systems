import pandas as pd
import numpy as np
import math
import random
import time

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
    #computing utlity matrix
    for i in range(num_of_ratings):
        user_id = ratings_list[i][0]
        movie_id = ratings_list[i][1]
        rating = ratings_list[i][2]
        user_movie_matrix[user_id][movie_id] = rating

    matrix_centered_zero = np.copy(user_movie_matrix)
    #center the test data set about the mean
    mean = 0.0
    for i in range(1,num_of_users):
        sum = 0.0
        count = 0.0
        #compute mean
        for j in  range(1,num_of_movies):
            if(user_movie_matrix[i][j] != 0):
                sum = sum + user_movie_matrix[i][j]
                count = count + 1.0
        mean = sum / count 
        #update data about mean
        for j in range(1,num_of_movies):
            if(user_movie_matrix[i][j] == 0.0):
                matrix_centered_zero[i][j] = mean
            else:
                matrix_centered_zero[i][j] = matrix_centered_zero[i][j] - mean

    #compute the training data by removing the values in the 1000*1000 matrix
    test = np.copy(matrix_centered_zero)
    for i in range(1,1001):
        for j in range(1,1001):
            if(matrix_centered_zero[i][j] != 0):
                test[i][j] = -1
    
    #recenter the training data about mean
    mean = 0.0
    for i in range(1,num_of_users):
        sum = 0.0
        count = 0.0
        #compute mean
        for j in  range(1,num_of_movies):
            if(test[i][j] == -1):
                sum = sum + 0.0
                count = count + 1.0
            elif(test[i][j] > 0):
                sum = sum + test[i][j]
                count = count + 1.0
        mean = sum / count 
        #recenter the data about mean
        for j in range(1,num_of_movies):
            if(test[i][j] == -1 or test[i][j] == 0):
                test[i][j] = mean
            else:
                test[i][j] = test[i][j] - mean
  
    #k factor of CUR decomposition
    k = 250

    #calculate the total sum of the sqaures of the elements
    total_sum_sq = 0.0
    for i in range(1,num_of_users):
        for j in range(1,num_of_movies):
            total_sum_sq = total_sum_sq + (test[i][j])**2

    #calculating the probabilty distribution for all the columns
    col_dis_pr = []
    col_dis_pr.append(0.0)
    for i in range(1,num_of_movies):
        col_sum_sq = 0.0
        for j in range(1,num_of_users):
            col_sum_sq = col_sum_sq + (test[j][i])**2
        col_dis_pr.append(col_sum_sq / total_sum_sq)

    #calculating the probabilty distribution for all the rows
    row_dis_pr = []
    row_dis_pr.append(0.0)
    for i in range(1,num_of_users):
        row_sum_sq = 0.0
        for j in range(1,num_of_movies):
            row_sum_sq = row_sum_sq + (test[i][j])**2
        row_dis_pr.append(row_sum_sq / total_sum_sq)

    #making a list with the indices of all columns
    cols_index = []
    cols_index.append(-1)
    for i in range(0,3952):
        cols_index.append(i+1)

    #making a list with the indices of all rows
    rows_index = []
    rows_index.append(-1)
    for i in range(0,6040):
        rows_index.append(i+1)

    #computing random values with given probability distribution for the rows and columns
    cols = np.random.choice(cols_index, 4 * k,replace=False, p = col_dis_pr)
    rows = np.random.choice(rows_index, 4 * k,replace=False, p = row_dis_pr)
    #c = 4 * k
    c_attr = 1000.0

    C = np.zeros((num_of_users, 4*k + 1))
    #computing the C matrix
    for i in range(1,4*k+1):
        C[:,i] = np.divide(test[:,cols[i-1]], np.sqrt(np.multiply(c_attr,col_dis_pr[cols[i-1]])))

    R = np.zeros((4*k+1, num_of_movies))
    #Computing the R matrix
    for i in range(1,4*k+1):
        R[i,:] = np.divide(test[rows[i-1],:], np.sqrt(np.multiply(c_attr,row_dis_pr[rows[i-1]])))
    #Computing the inverses
    C_inv = np.linalg.pinv(C)
    R_inv = np.linalg.pinv(R)
    #Computing U
    U = np.matmul(np.matmul(C_inv, test), R_inv)
    #Computing the original matrix from C , U , R
    answer = np.matmul(np.matmul(C, U), R)

    squares_sum = 0.0
    count_sq = 0.0
    precision_rating = []
    for i in range(1,1001):
        for j in range(1,1001):
            if(user_movie_matrix[i][j] != 0):
                precision_rating.append(answer[i][j])
                print("Actual rating")
                print(user_movie_matrix[i][j])
                print("Predicted rating")
                print(answer[i][j])
                #compute the rmse
                squares_sum = squares_sum + (answer[i][j] - user_movie_matrix[i][j])**2
                count_sq = count_sq + 1.0
                print("Root mean squared error")
                print(math.sqrt(squares_sum / count_sq))
                if(count_sq > 1):
                    print("Spearman's correlation")
                    correlation = 1 - ((6 * squares_sum) / (count_sq**3 - count_sq))
                    print(correlation)
                print("")
    #calculation of precision at top k
    precision_rating.sort(reverse=True)
    countk = 0.0
    for i in range(0, precision_k):
        if(precision_rating[i] >= 3.0):
            countk = countk + 1
    precision_at_topk = countk / precision_k
    print("Precision at top k")
    print(precision_at_topk)

if __name__== "__main__":
    main()

