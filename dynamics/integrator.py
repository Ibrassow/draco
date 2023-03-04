

# Runge-Kutta 4

def rk4(dynamics, x, u, dt):
    k1 = dt * dynamics(x, u)
    k2 = dt * dynamics(x + k1 * 0.5, u)
    k3 = dt * dynamics(x + k2 * 0.5, u)
    k4 = dt * dynamics(x + k3, u)
    x = x + (1/6) * (k1 + 2 * k2 + 2 * k3 + k4)
    
    return x