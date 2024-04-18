def rk4(f, x0, y0, dt, mu):

    k1 = f(x0, y0, mu)
    k2 = f(x0 + dt / 2, y0 + dt / 2 * k1[0], mu)
    k3 = f(x0 + dt / 2, y0 + dt / 2 * k2[0], mu)
    k4 = f(x0 + dt, y0 + dt * k3[0], mu)

    x_new = x0 + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
    y_new = y0 + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])

    return [x_new, y_new]
