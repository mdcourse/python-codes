

def potentials(epsilon, sigma, r, derivative=False):
    if derivative:
        return 48 * epsilon * ((sigma / r) ** 12 - 0.5 * (sigma / r) ** 6) / r
    else:
        return 4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)

