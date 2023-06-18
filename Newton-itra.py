import numpy as np
import pandas as pd

# Import the Netflix dataset from the provided link
url = "Dataset_Test.csv"
data = pd.read_csv(url)

# Split the data into train and test sets
train_data = data.iloc[:5000]
test_data = data.iloc[5000:]

# Create a user-movie rating matrix from the train data
users = train_data["user_id"].unique()
movies = train_data["Movie_Name"].unique()

n_users = len(users)
n_movies = len(movies)

# Create a dictionary to map users and movies to their respective indices
user_indices = {user: index for index, user in enumerate(users)}
movie_indices = {movie: index for index, movie in enumerate(movies)}

ratings = np.zeros((n_users, n_movies))

for row in train_data.itertuples():
    user_index = user_indices.get(row[1])
    movie_index = movie_indices.get(row[2])
    if user_index is not None and movie_index is not None:
        ratings[user_index, movie_index] = row[3]

# Define the mean squared error function
def mse(x, y):
    return np.mean(np.square(x - y))

# Define the user-based similarity matrix
def similarity_matrix(ratings):
    # Normalize the ratings by subtracting the user mean
    user_mean = np.mean(ratings, axis=1, keepdims=True)
    ratings_norm = ratings - user_mean

    # Compute the cosine similarity between users
    dot_product = np.dot(ratings_norm, ratings_norm.T)
    norm = np.linalg.norm(ratings_norm, axis=1, keepdims=True)
    similarity = dot_product / ((norm * norm.T)+ 1e-8)

    # Return the similarity matrix
    return similarity

# Compute the user-based similarity matrix from the ratings
similarity = similarity_matrix(ratings)

# Define the user-based prediction function
def predict(ratings, similarity):
    # Compute the weighted average of ratings using similarity as weights
    weighted_sum = np.dot(similarity, ratings)
    sum_of_weights = np.sum(np.abs(similarity), axis=1, keepdims=True)
    prediction = weighted_sum / (sum_of_weights+ 1e-8)

    # Return the prediction matrix
    return prediction

# Compute the initial prediction from the ratings and similarity
prediction = predict(ratings, similarity)

# Define the objective function to minimize using Newton's method
def objective(similarity):
    # Compute the prediction from the similarity matrix
    prediction = predict(ratings, similarity)

    # Compute the mean squared error between prediction and ratings
    error = mse(prediction, ratings)

    # Return the error
    return error

# Define the gradient of the objective function
def gradient(similarity):
    # Compute the prediction from the similarity matrix
    prediction = predict(ratings, similarity)

    # Compute the gradient using the formula
    error = prediction - ratings
    grad = np.dot(error, ratings.T) / (np.sum(np.abs(similarity), axis=1, keepdims=True)+ 1e-8)

    # Return the gradient
    return grad

# Define the Hessian of the objective function
def hessian(similarity):
    # Compute the prediction from the similarity matrix
    prediction = predict(ratings, similarity)

    # Compute the error and its derivative
    error = prediction - ratings
    error_deriv = np.dot(error, ratings.T)

    # Compute the Hessian using the formula
    hess = -2 * np.dot(error_deriv.T * similarity, similarity.T) / (np.sum(np.abs(similarity), axis=1, keepdims=True)+ 1e-8)

    # Return the Hessian
    return hess

def newton_method(similarity, max_iter=10, tol=1e-6):
    # Initialize a counter for iterations
    i = 0

    # Initialize a list to store the objective values at each iteration
    objective_values = []

    # Loop until convergence or maximum iterations
    while i < max_iter:
        # Compute the current objective value, gradient, and Hessian
        f = objective(similarity)
        grad = gradient(similarity)
        hess = hessian(similarity)

        # Check if the gradient is close to zero
        if np.linalg.norm(grad) < tol:
            break

        # Update the similarity matrix using the Newton's update rule
        similarity -= np.linalg.inv(hess) @ grad

        # Store the objective value at the current iteration
        objective_values.append(f)

        # Increment the counter
        i += 1

        # Print the iteration number and the objective value at each iteration
        print("Iteration", i, "- Objective value:", f)

    # Return the optimized similarity matrix, the final objective value, and the objective values at each iteration
    return similarity, f, objective_values

# Run Newton's method on the initial similarity matrix
similarity_opt, f_opt = newton_method(similarity)

# Print the final objective value and the optimized similarity matrix
print("Final objective value:", f_opt)
print("Optimized similarity matrix:")
print(similarity_opt)

# Use the optimized similarity matrix to predict ratings for the test data
test_ratings = np.zeros((n_users, n_movies))
for row in test_data.itertuples():
    user_index = user_indices.get(row[1])
    movie_index = movie_indices.get(row[2])
    if user_index is not None and movie_index is not None:
        test_ratings[user_index, movie_index] = row[3]

test_prediction = predict(test_ratings, similarity_opt)

# Calculate the mean squared error (MSE) and root mean squared error (RMSE)
mse_test = mse(test_prediction, test_ratings)
rmse_test = np.sqrt(mse_test)

print("MSE on test data:", mse_test)
print("RMSE on test data:", rmse_test)
