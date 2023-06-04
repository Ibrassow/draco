

# Runge-Kutta 4

def rk4(dynamics, x, u, dt):
    k1 = dt * dynamics(x, u)
    k2 = dt * dynamics(x + k1 * 0.5, u)
    k3 = dt * dynamics(x + k2 * 0.5, u)
    k4 = dt * dynamics(x + k3, u)
    x = x + (1/6) * (k1 + 2 * k2 + 2 * k3 + k4)
    
    return x



def hermite_simpson(dynamics, x1, x2, u, dt):
    # hermite simpson implicit integrator residual
    x_k12 = 0.5 * (x1 + x2) + (dt / 8) * (dynamics(x1, u) - dynamics(x2, u))
    return x1 + (dt / 6) * (dynamics(x1, u) + 4 * dynamics(x_k12, u) + dynamics(x2, u)) - x2


def implicit_midpoint(dynamics, x1, x2, u, dt):
    # implicit midpoint integrator residual
    return x1 + dt * dynamics(0.5 * (x1 + x2), u) - x2