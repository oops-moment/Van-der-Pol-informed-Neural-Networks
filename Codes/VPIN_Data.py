import numpy as np


def vanderpol(x, y, mu):
    """
  Function representing the Van der Pol oscillator.

  Args:
      x: Current position.
      y: Current velocity.
      mu: Parameter controlling nonlinearity.

  Returns:
      A tuple containing the rate of change of position and velocity.
  """
    return (y, mu * (1 - x**2) * y - x)


def rk4(f, x0, y0, dt, mu):
    """
  Runge-Kutta method (RK4) for solving differential equations.

  Args:
      f: Function representing the differential equation.
      x0: Initial position.
      y0: Initial velocity.
      dt: Time step.
      mu: Parameter controlling nonlinearity.

  Returns:
      A list of positions and velocities at each time step.
  """
    k1 = f(x0, y0, mu)
    k2 = f(x0 + dt / 2, y0 + dt / 2 * k1[0], mu)
    k3 = f(x0 + dt / 2, y0 + dt / 2 * k2[0], mu)
    k4 = f(x0 + dt, y0 + dt * k3[0], mu)

    x_new = x0 + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
    y_new = y0 + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])

    return [x_new, y_new]


# Define parameters
mu = 4  # Parameter for nonlinearity
t_start = 0  # Starting time
t_end = 10  # End time
dt = 0.04  # Time step
x0 = 1  # Initial position
y0 = 0  # Initial velocity

# Initialize lists for storing data
t = []
x = []
y = []

# Simulate using RK4
t.append(t_start)
x.append(x0)
y.append(y0)

for i in range(int((t_end - t_start) / dt)):
    next_point = rk4(vanderpol, x[-1], y[-1], dt, mu)
    t.append(t[-1] + dt)
    x.append(next_point[0])
    y.append(next_point[1])

# Print or use the generated data points (t, x, y)
print("Time:", t)
print("Position:", x)
print("Velocity:", y)
