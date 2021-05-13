import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

for i in range(N-1):
    t=[tau[i], tau[i+1]]
    sol = odeint(move_func, s0, t)
    X.append(sol[1, 0])
    Y.append(sol[1, 1])
    x0 = sol[1, 0]
    y0 = sol[1, 1]
    vx0 = sol[1, 2]
    vy0 = sol[1, 3]
    if np.abs(y0-Y1) <= radius or np.abs(y0-Y2) <= radius:
        vy0 = -vy0
    s0 = x0, y0, vx0, vy0
