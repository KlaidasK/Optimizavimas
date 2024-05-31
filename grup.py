import numpy as np
import matplotlib.pyplot as plt

# Constants and functions as defined previously
W = 500000  # Constant in the equality constraint

def objective_function(x):
    return 10*x[0]**2 + 5*x[1]**2 + 3*x[2]**2

def constraint1(x):
    return 100 * x[0] + 50 * x[1] + 30 * x[2] - W

def constraint2(x):
    return x[1] + x[2] - x[0]

def grad_augmented_lagrange(x, lambdas, rho):
    grad_obj = np.array([20*x[0], 10*x[1], 6*x[2]])
    grad_con1 = np.array([100, 50, 30])
    grad_con2 = np.array([-1, 1, 1])
    penalty_grad1 = rho * constraint1(x) * grad_con1
    penalty_grad2 = rho * constraint2(x) * grad_con2
    return grad_obj + lambdas[0] * grad_con1 + lambdas[1] * grad_con2 + penalty_grad1 + penalty_grad2

def gradient_descent(x_init, lambdas_init, rho, learning_rate, num_iterations):
    x = x_init
    lambdas = lambdas_init
    f_values = []
    x_values = []  # Store optimal x values for each iteration
    for _ in range(num_iterations):
        grad = grad_augmented_lagrange(x, lambdas, rho)
        x = x - learning_rate * grad
        lambdas[0] = lambdas[0] + learning_rate * rho * constraint1(x)
        lambdas[1] = lambdas[1] + learning_rate * rho * constraint2(x)
        f_values.append(objective_function(x))
        x_values.append(x.copy())
    return x, lambdas, f_values, x_values

# Parameters for the gradient descent
x_init = np.array([1, 1, 1])
lambdas_init = np.array([1, 1])
rho = 1.0  # Penalty parameter
learning_rate = 0.00002
num_iterations = 1000

# Perform gradient descent
x_opt, lambdas_opt, f_values, x_values = gradient_descent(x_init, lambdas_init, rho, learning_rate, num_iterations)
fx_opt = objective_function(x_opt)


print("Optimal x,y,z: ", x_opt)
print("Optimal f(x): ", fx_opt)
print("Optimal lambdas:", lambdas_opt)
print("Step count:", num_iterations)

# Plotting the objective function values
plt.figure(figsize=(10, 6))
plt.plot(f_values, label='Objective Function')
plt.scatter(0, objective_function(x_init), color='red', label='Initial Point')  # Initial point
plt.scatter(len(f_values)-1, fx_opt, color='green', label='Optimal Point')  # Optimal point
plt.title('Objective Function Value During Gradient Descent')
plt.xlabel('Iteration')
plt.ylabel('Objective Function Value')
plt.legend()
plt.grid(True)
plt.show()


# Plotting the optimal x values
plt.figure(figsize=(10, 6))
for i in range(3):
    plt.plot([x[i] for x in x_values], label='x{}'.format(i+1))
plt.title('Optimal x Values During Gradient Descent')
plt.xlabel('Iteration')
plt.ylabel('x Value')
plt.legend()
plt.grid(True)
plt.show()