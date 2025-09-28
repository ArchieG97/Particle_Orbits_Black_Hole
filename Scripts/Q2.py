import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
import os

# Directory for saving figures
output_dir = r"C:\CATAM_BlackHoleOrbits\Figures_BH"
os.makedirs(output_dir, exist_ok=True)

# Parameters
Hs = [0.0, 0.5, 1.0, 1.5, 2.0]  # angular momentum values
epsilon = 1
r0 = 6
k = np.sqrt(25 / 27)  # same energy constant for all

ds = 0.01  # step size
max_steps = int(200 * np.pi / ds)  # maximum steps for plunging orbits

cycle_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Schwarzschild metric function
def F(r):
    return 1 - 1/r

# Effective potential derivative for a given h
def dVeff_dr(r, h):
    return -2*h**2 / r**3 + 1 / r**2 + 3*h**2 / r**4

# ODE system: y = [r, r_dot, phi, s]
def derivatives(y, h):
    r, r_dot, phi, s = y
    dr_ds = r_dot
    d2r_ds = -0.5 * dVeff_dr(r, h)
    dphi_ds = h / r**2
    ds_ds = 1.0
    return np.array([dr_ds, d2r_ds, dphi_ds, ds_ds])

# Integrator
def integrate_orbit(h):
    y = np.array([r0, 0.0, 0.0, 0.0])
    traj = []
    if h == 2.0:
        period = 2 * np.pi / (h / r0**2)
        steps = int(np.ceil(period / ds)) + 1
    else:
        steps = max_steps
    for _ in range(steps):
        r = y[0]
        traj.append(y.copy())
        if h != 2.0 and r <= 1.0:
            break
        k1 = derivatives(y, h)
        k2 = derivatives(y + 0.5 * ds * k1, h)
        k3 = derivatives(y + 0.5 * ds * k2, h)
        k4 = derivatives(y + ds * k3, h)
        y = y + (ds / 6) * (k1 + 2*k2 + 2*k3 + k4)
    return np.array(traj)

# Storage for capture analysis
captured = []
capture_times = {}

# Cartesian plot with fading trajectories
fig1, ax1 = plt.subplots(figsize=(6,6))
handles = []

for idx, h in enumerate(Hs):
    traj = integrate_orbit(h)
    if traj[-1,0] <= 1.0 and h != 2.0:
        captured.append(h)
        capture_times[h] = traj[-1,3]
    r_vals = traj[:,0]
    phi_vals = traj[:,2]
    x_vals = r_vals * np.cos(phi_vals)
    y_vals = r_vals * np.sin(phi_vals)
    base_color = cycle_colors[idx % len(cycle_colors)]
    if h == 2.0:
        label = r"$h_{\mathrm{cir}} = %.1f$" % h
        line, = ax1.plot(x_vals, y_vals, color=base_color, label=label)
        handles.append(line)
    else:
        rgba_base = mcolors.to_rgba(base_color)
        alphas = np.linspace(1.0, 0.01, len(x_vals))
        colors_rgba = [(rgba_base[0], rgba_base[1], rgba_base[2], a) for a in alphas]
        ax1.scatter(x_vals, y_vals, color=colors_rgba, s=2)
        label = r"$h = %.1f$" % h
        handles.append(Line2D([0],[0], color=base_color, label=label))

# Start point marker
start_handle, = ax1.plot(r0, 0, marker='x', color='black', markersize=10, linestyle='None', label=r"Start Point")
handles.append(start_handle)

# Black hole marker
black_hole_patch = plt.Circle((0,0),1,color='black')
ax1.add_patch(black_hole_patch)
black_hole_handle = Line2D([0],[0],marker='o',color='black',linestyle='None',markersize=10,label=r"Black Hole")
handles.append(black_hole_handle)

ax1.legend(handles=handles, loc='upper right')
ax1.set_aspect('equal')
ax1.set_xlim(-6.5,6.5)
ax1.set_ylim(-6.5,6.5)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Particle Orbits around a Black Hole')
ax1.grid(True)
plt.savefig(os.path.join(output_dir, 'Q2_Orbits_xy.pdf'), dpi=300, bbox_inches='tight')

# r-phi plot
fig2, ax2 = plt.subplots(figsize=(6,4))
# Shade region
ax2.axhspan(0, 1, color='lightgrey', alpha=0.5)
for h in Hs:
    traj = integrate_orbit(h)
    r_vals, phi_vals = traj[:,0], traj[:,2]
    if h == 2.0:
        label = r"$h_{\mathrm{cir}} = %.1f$" % h
    else:
        label = r"$h = %.1f$" % h
    ax2.plot(phi_vals, r_vals, label=label)
ax2.axhline(1, color='black', linestyle='--')
ax2.set_ylim(bottom=0)
ax2.set_xlabel(r'$\phi$')
ax2.set_ylabel('r')
ax2.set_title('Radius of Orbit as a Function of Angle')
ax2.legend(loc='upper right')
ax2.grid(True)
plt.savefig(os.path.join(output_dir, 'Q2_Orbits_rphi.pdf'), dpi=300, bbox_inches='tight')

# Print capture results
print("Captured angular momenta:", captured)
for h in captured:
    print(f"h = {h} captured in proper time s = {capture_times[h]:.2f}")

plt.show()
