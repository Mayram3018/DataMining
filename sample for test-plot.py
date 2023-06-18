import numpy as np
import matplotlib.pyplot as plt

# Sample dataset
users = ['User1', 'User2', 'User3']
movies = ['Movie1', 'Movie2', 'Movie3']

# Ratings matrix
ratings = np.array([[3, 4, 0],
                    [0, 5, 2],
                    [1, 0, 4]])

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
    similarity = dot_product / ((norm * norm.T) + 1e-8)

    # Return the similarity matrix
    return similarity

# Define the user-based prediction function
def predict(ratings, similarity):
    # Compute the weighted average of ratings using similarity as weights
    weighted_sum = np.dot(similarity, ratings)
    sum_of_weights = np.sum(np.abs(similarity), axis=1, keepdims=True)
    prediction = weighted_sum / (sum_of_weights + 1e-8)

    # Return the prediction matrix
    return prediction

# Define the objective function for Newton's method
def objective_newton(similarity):
    # Compute the prediction from the similarity matrix
    prediction = predict(ratings, similarity)

    # Compute the mean squared error between prediction and ratings
    error = mse(prediction, ratings)

    # Return the error
    return error

# Define the objective function for AdaGrad
def objective_adagrad(similarity):
    # Compute the prediction from the similarity matrix
    prediction = predict(ratings, similarity)

    # Compute the mean squared error between prediction and ratings
    error = mse(prediction, ratings)

    # Return the error
    return error

# Implement Newton's method to optimize the similarity matrix
def newton_method(similarity, max_iter=10, tol=1e-6):
    # Initialize a counter for iterations
    i = 0

    # Initialize a list to store objective values
    objective_values = []

    # Loop until convergence or maximum iterations
    while i < max_iter:
        # Compute the current objective value
        f = objective_newton(similarity)

        # Store the objective value
        objective_values.append(f)

        # Print the iteration number and objective value
        print("Newton's method - Iteration:", i+1, "Objective value:", f)

        # Rest of the code...

        # Increment the counter
        i += 1

    # Plot the objective values over iterations
    plt.plot(range(1, len(objective_values) + 1), objective_values)
    plt.xlabel("Iteration")
    plt.ylabel("Objective Value")
    plt.title("Objective Value Convergence (Newton's Method)")
    plt.show()

# Implement AdaGrad method to optimize the similarity matrix
def adagrad_method(similarity, max_iter=10, tol=1e-6, learning_rate=0.01):
    # Initialize a counter for iterations
    i = 0

    # Initialize a list to store objective values
    objective_values = []

    # Loop until convergence or maximum iterations
    while i < max_iter:
        # Compute the current objective value
        f = objective_adagrad(similarity)

        # Store the objective value
        objective_values.append(f)

        # Print the iteration number and objective value
        print("AdaGrad method - Iteration:", i+1, "Objective value:", f)

        # Rest of the code...

        # Increment the counter
        i += 1

    # Plot the objective values over iterations
    plt.plot(range(1, len(objective_values) + 1), objective_values)
    plt.xlabel("Iteration")
    plt.ylabel("Objective Value")
    plt.title("Objective Value Convergence (AdaGrad Method)")
    plt.show()

# Compute the user-based similarity matrix from the ratings
similarity = similarity_matrix(ratings)

# Print the initial similarity matrix
print("Initial similarity matrix:")
print(similarity)
print()

# Run Newton's method on the initial similarity matrix
newton_method(similarity)

# Run AdaGrad method on the initial similarity matrix
adagrad_method(similarity)
