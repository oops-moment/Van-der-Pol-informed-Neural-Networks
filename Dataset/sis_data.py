import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# Define the SIS model equations
def f(t, r):
    S, I = r
    dSdt = -beta * S * I + gamma * I
    dIdt = beta * S * I - gamma * I
    return [dSdt, dIdt]


# Parameters
beta = 0.2  # infection rate
gamma = 0.1  # recovery rate

# Initial conditions
r0 = [0.99, 0.01]  # 99% susceptible, 1% infected at t=0

# Time span
t_span = [0, 160]

# Solve using RK4
sol = solve_ivp(f, t_span, r0, t_eval=np.linspace(t_span[0], t_span[1], 5000))

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot(sol.t, sol.y[0], label='S')
plt.plot(sol.t, sol.y[1], label='I')
plt.xlabel('Time')
plt.ylabel('State Variables')
plt.title('SIS Model Solution using RK4')
plt.legend()
plt.grid(True)
plt.show()

np.savetxt('../Dataset/sis_data.dat',
           np.column_stack((sol.y[0], sol.y[1])),
           header='S I',
           fmt='%12.6f')
