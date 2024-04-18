def lorenz(t, r):
    sigma = 10
    rho = 28
    beta = 8/3
    x, y, z = r
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]
