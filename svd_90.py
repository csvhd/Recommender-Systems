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

    #computing the utility matrix
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
        #computing the mean
        for j in  range(1,num_of_movies):
            if(user_movie_matrix[i][j] != 0):
                sum = sum + user_movie_matrix[i][j]
                count = count + 1.0
        mean = sum / count 
        #update the values with the mean
        for j in range(1,num_of_movies):
            if(user_movie_matrix[i][j] == 0.0):
                matrix_centered_zero[i][j] = mean
            else:
                matrix_centered_zero[i][j] = matrix_centered_zero[i][j] - mean
    #make the training data set, with missing values as -1
    test = np.copy(matrix_centered_zero)
    for i in range(1,1001):
        for j in range(1,1001):
            if(matrix_centered_zero[i][j] != 0):
                test[i][j] = -1
    #center the training data
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
        if(i == 1):
            print(sum)
            print(count)   
            print(mean)
            #center around mean
        for j in range(1,num_of_movies):
            if(test[i][j] == -1 or test[i][j] == 0):
                test[i][j] = mean
            else:
                test[i][j] = test[i][j] - mean
    #Compute A transpose multiplied A        
    AtA = np.dot(np.transpose(test), test)
    #compute the eigen vectors and values
    eigen_values_V , eigen_vectors_V = LA.eig(AtA)
    #discard the imaginary part of the eigen values and vectors
    eigen_values_V = np.real(eigen_values_V)
    eigen_vectors_V = np.real(eigen_vectors_V)
    #sort the eigen values in descending order
    idV = np.argsort(np.multiply(-1,eigen_values_V))
    eigen_values_V = eigen_values_V[idV]
    #rearrange the eigen vectors as per the eigen values
    eigen_vectors_V = eigen_vectors_V[:, idV]
    #compute the sigma matrix
    S = np.sqrt(np.abs(eigen_values_V))
    S = np.diag(S)
    #compute the inverse of the sigma matrix
    Sinv = np.linalg.pinv(S)
    #Compute the left SVD matrix
    U = np.matmul(np.matmul(test, eigen_vectors_V), Sinv)
    #transpose the right SVD matrix
    eigen_vectors_V = np.transpose(eigen_vectors_V)

    energy = 0.0
    #compute total energy
    for i in range(S.shape[0]):
        energy = energy + S[i][i]**2
    #compute 90% of the total
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
    new_U = U[:,0:new_shape]
    eigen_vectors_V = eigen_vectors_V[0:new_shape, :]
    #compute the original matrix by multiplying all three but with 90% energy
    answer = np.matmul(np.matmul(new_U, new_S), (eigen_vectors_V))
    squares_sum = 0.0
    count_sq = 0.0
    precision_rating = []
    for i in range(1,1001):
        for j in range(1,1001):
            if(matrix_centered_zero[i][j] != 0):
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
    #computing the precision at top k
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