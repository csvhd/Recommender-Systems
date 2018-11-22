import pandas as pd
import numpy as np
import math
import time

precision_k = 458
num_of_users = 6040 + 1
num_of_movies= 3952 + 1
num_of_ratings = 1000209

def main():
    #Reading ratings file:
    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    ratings = pd.read_csv('ml-1m/ratings.dat', sep="::", names=r_cols,encoding='latin-1',engine='python')
    #removing the timestamp
    ratings = ratings[['user_id', 'movie_id', 'rating']]
    #converting to list
    ratings_list = ratings.values.tolist()
    user_movie_matrix = np.zeros((num_of_users,num_of_movies))
    #making the utlity matrix
    for i in range(num_of_ratings):
        user_id = ratings_list[i][0]
        movie_id = ratings_list[i][1]
        rating = ratings_list[i][2]
        user_movie_matrix[user_id][movie_id] = rating
    user_movie_matrix = np.transpose(user_movie_matrix)
    #Updating the top 100 * 100 matrix data to zero to use later for testing
    matrix_without_test_data = np.copy(user_movie_matrix)
    for i in range(1,101):
        for j in range(1,101):
            matrix_without_test_data[i][j] = 0.0
    #centering the matrix around mean
    for i in range(1,num_of_movies):
        sum = 0.0
        count = 0.0
        for j in  range(1,num_of_users):
            #calculating the mean of each row
            if(matrix_without_test_data[i][j] != 0):
                sum = sum + matrix_without_test_data[i][j]
                count = count + 1.0
        if(count == 0):
            continue
        mean = sum / count    
        for j in range(1,num_of_users):
            #centering the data
            if(matrix_without_test_data[i][j] != 0):
                matrix_without_test_data[i][j] = matrix_without_test_data[i][j] - mean
            else:
                matrix_without_test_data[i][j] = mean
    similarity = 0.0
    predict = 0.0
    count = 0.0
    squares_sum = 0.0
    count_sq = 0.0
    #start of prediction
    start = time.time()
    precision_rating = []
    for a in range(1,101):
        for b in range(1,101):
            if(user_movie_matrix[a][b] != 0):
                #taking individual columns of users
                col_A = matrix_without_test_data[:,b]
                dot_products = np.zeros((num_of_users,1))
                for k in range(1, num_of_users):
                    if(matrix_without_test_data[a][k] != 0):
                        #taking the rest of the user vectors
                        col_B = matrix_without_test_data[:,k]
                        A = np.sqrt(np.sum(col_A**2))
                        B =  np.sqrt(np.sum(col_B**2))   
                        if(A == 0 or B == 0):
                            similarity = 0.0
                        else:
                            #Computing their cosine similarity
                            similarity = (np.sum(np.multiply(col_A,col_B))) / (np.sqrt(np.sum(col_A**2)) * np.sqrt(np.sum(col_B**2)))
                        dot_products[k][0] = similarity
                predict = 0.0
                count = 0.0
                for k in range(1,num_of_users):
                    #predicting the value with cosine similarity
                    if(matrix_without_test_data[a][k] != 0 and dot_products[k][0] > 0):
                        predict =  predict + dot_products[k][0] * matrix_without_test_data[a][b]
                        count = count + dot_products[k][0]
                if(count > 0):
                    temp = predict
                    predict = predict / count
                precision_rating.append(predict)
                print("Predicted Rating ")
                print(predict)     
                print("Actual Rating ")
                print(user_movie_matrix[a][b])
                #computing root mean squared error  
                squares_sum = squares_sum + (predict - user_movie_matrix[a][b])**2
                count_sq = count_sq + 1.0
                print("Root mean squared error")
                print(math.sqrt(squares_sum / count_sq))
                if(count_sq > 1):
                    print("Spearman's correlation")
                    correlation = 1 - ((6 * squares_sum) / (count_sq**3 - count_sq))
                    print(correlation)
                print("")
    print(count_sq)
    #computing the precision at top k
    precision_rating.sort(reverse=True)
    countk = 0.0
    for i in range(0,precision_k):
        if(precision_rating[i] >= 3):
            countk = countk + 1
    precision_at_topk = countk / precision_k
    print("Precision at top k")
    print(precision_at_topk)

if __name__ == "__main__":
    main()


