from re import X
from leastsquares import leastsq_cpp

Xs = 914250.0
Ys = 575400.0
Zs = 800.0
phi = 0.0
omega = 0.0
kappa = -1.57
m = 15000.0
f = 0.152222

# 4 points xyXYZ
points = "56.515 -78.969 913928.64 575198.44 189.64 1.242 1.134 914270.77 575432.35 191.26 95.576 97.171 914684.64 575022.09 186.72 -70.988 92.733 914662.47 575738.30 191.94"

h = leastsq_cpp(points, Xs, Ys, Zs, phi, omega, kappa, m, f)

print(h)