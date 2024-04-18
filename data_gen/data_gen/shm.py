import numpy as np

def shm(mass, spring_constant, initial_position, initial_velocity, time_step, num_steps):

  omega = np.sqrt(spring_constant / mass)  # Angular frequency

  time = np.linspace(0, time_step * num_steps, num_steps + 1)
  position = np.zeros(num_steps + 1)
  velocity = np.zeros(num_steps + 1)

  position[0] = initial_position
  velocity[0] = initial_velocity

  for i in range(1, num_steps + 1):
    k1_p = velocity[i - 1]
    k1_v = -omega**2 * position[i - 1]

    k2_p = velocity[i - 1] + 0.5 * time_step * k1_v
    k2_v = -omega**2 * (position[i - 1] + 0.5 * time_step * k1_p)

    k3_p = velocity[i - 1] + 0.5 * time_step * k2_v
    k3_v = -omega**2 * (position[i - 1] + 0.5 * time_step * k2_p)

    k4_p = velocity[i - 1] + time_step * k3_v
    k4_v = -omega**2 * (position[i - 1] + time_step * k3_p)

    position[i] = position[i - 1] + time_step * (k1_p + 2 * k2_p + 2 * k3_p + k4_p) / 6
    velocity[i] = velocity[i - 1] + time_step * (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6

  return time, position, velocity

