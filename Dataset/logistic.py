import numpy as np
import matplotlib.pyplot as plt

# Define the logistic function
def logistic(r, x):
    return r * x * (1 - x)

# Parameters
r_values = np.linspace(2.5, 4.0, 10000)  # range of r values
x = 0.5  # initial condition

# Time steps
t_steps = 1000
last_t_steps = 100  # only plot the last t_steps iterations to remove transient

# Initialize array to store values
x_values = np.empty((len(r_values), last_t_steps))

# Generate data
for i, r in enumerate(r_values):
    for t in range(t_steps):
        if t >= t_steps - last_t_steps:
            x_values[i, t + last_t_steps - t_steps] = x
        x = logistic(r, x)

# Plot the results
plt.figure(figsize=(8, 6))
for i in range(last_t_steps):
    plt.scatter(r_values, x_values[:, i], s=0.1, color='k')
plt.xlabel('r')
plt.ylabel('x')
# plt.title('Bifurcation Diagram of the Logistic Map')
plt.grid(True)
plt.show()
