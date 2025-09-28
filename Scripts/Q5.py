import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.collections import LineCollection
import os

# Output directory
output_dir = r"C:\CATAM_BlackHoleOrbits\Figures_BH"
os.makedirs(output_dir, exist_ok=True)

# Parameters
Hs = [0.0, 0.5, 1.0, 1.5, 2.0]
r0 = 6
k = np.sqrt(25 / 27)
ds = 0.01
t_max = 200
dt = 0.01
t_eval = np.arange(0, t_max, dt)
max_steps = int(300 * np.pi / ds)

def dVeff_dr(r, h):
    return -2*h**2 / r**3 + 1 / r**2 + 3*h**2 / r**4

def derivatives_affine(y, h):
    r, r_dot, phi, tau = y
    dr_ds = r_dot
    d2r_ds = -0.5 * dVeff_dr(r, h)
    dphi_ds = h / r**2
    return np.array([dr_ds, d2r_ds, dphi_ds, 1.0])

def integrate_affine_orbit(h):
    y = np.array([r0, 0.0, 0.0, 0.0])
    traj = []
    for _ in range(max_steps):
        r = y[0]
        traj.append(y.copy())
        if h != 2.0 and r <= 1.01:
            break
        k1 = derivatives_affine(y, h)
        k2 = derivatives_affine(y + 0.5 * ds * k1, h)
        k3 = derivatives_affine(y + 0.5 * ds * k2, h)
        k4 = derivatives_affine(y + ds * k3, h)
        y = y + (ds / 6) * (k1 + 2*k2 + 2*k3 + k4)
    return np.array(traj)

def f_prime(r, k, h):
    if r <= 1.01:
        return 1e-8  # small fallback to avoid early stop
    A = (r - 1)**2 / (k**2 * r**2)
    B = k**2 - (1 - 1/r) * (h**2 / r**2 + 1)
    dA = (2*(r - 1)*r**2 - (r - 1)**2*2*r) / (k**2 * r**4)
    term1 = (1 - 1/r)
    term2 = (h**2 + r**2) / r**2
    dterm2 = -2 * (h**2 + r**2) / r**3 + 2 / r
    dB = -dterm2 * term1 - term2 / r**2
    return dA * B + A * dB

def coordinate_time_rhs(t, y, k, h):
    r, r_dot = y
    return [r_dot, 0.5 * f_prime(r, k, h)]

def compute_phi_t(sol, h):
    r_vals = sol.y[0]
    phi = np.zeros_like(r_vals)
    for i in range(1, len(r_vals)):
        r2 = r_vals[i]**2
        if r2 < 1e-6:
            break
        dphi = h / r2 * (sol.t[i] - sol.t[i-1])
        phi[i] = phi[i-1] + dphi
    return phi

def fading_line(ax, x, y, color, label=None):
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    alpha_vals = np.linspace(1.0, 0.05, len(segments))
    lc = LineCollection(segments, colors=[(*color[:3], a) for a in alpha_vals], linewidths=1.5, label=label)
    ax.add_collection(lc)

# Run simulations
trajectories_affine = {}
trajectories_coord = {}
for h in Hs:
    trajectories_affine[h] = integrate_affine_orbit(h)
    sol = solve_ivp(coordinate_time_rhs, [0, t_max], [r0, 0.0],
                    args=(k, h), t_eval=t_eval, rtol=1e-8, atol=1e-10)
    trajectories_coord[h] = sol

# Radial Coordinate Plot
fig, (ax_tau, ax_t) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
color_list = plt.get_cmap('tab10').colors

for idx, h in enumerate(Hs):
    color = color_list[idx % len(color_list)]
    traj = trajectories_affine[h]
    sol = trajectories_coord[h]

    # Clip tau
    mask_tau = traj[:, 3] <= t_max
    ax_tau.plot(traj[mask_tau, 3], traj[mask_tau, 0], '-', color=color, label=fr"$h = {h:.1f}$")

    # Clip t
    mask_t = sol.t <= t_max
    ax_t.plot(sol.t[mask_t], sol.y[0][mask_t], '-', color=color)

for ax in [ax_tau, ax_t]:
    ax.axhline(1, color='black', linestyle='--')
    ax.set_ylabel(r"$r$")
    ax.grid(True)

ax_tau.set_title("Radial Coordinate vs Time")
ax_tau.set_ylabel(r"$r(\tau)$")
ax_tau.set_xlabel(r"$\tau$")
ax_t.set_ylabel(r"$r(t)$")
ax_t.set_xlabel(r"$t$")

handles, labels = ax_tau.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=len(Hs), fontsize=9)
plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(output_dir, 'Q5_Compare_r_vs_time.pdf'), dpi=300, bbox_inches='tight')
plt.show()

# Spatial Plot
fig_spatial, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

for idx, h in enumerate(Hs):
    color = color_list[idx % len(color_list)]

    # Proper-time
    traj = trajectories_affine[h]
    r_aff, phi_aff = traj[:, 0], traj[:, 2]
    x_aff = r_aff * np.cos(phi_aff)
    y_aff = r_aff * np.sin(phi_aff)
    fading_line(ax_left, x_aff, y_aff, color, label=fr"$h = {h:.1f}$")
    ax_left.plot(x_aff[0], y_aff[0], 'x', color='black', markersize=6)

    # Coordinate-time
    sol = trajectories_coord[h]
    r_coord = sol.y[0]
    phi_coord = compute_phi_t(sol, h)
    x_coord = r_coord * np.cos(phi_coord)
    y_coord = r_coord * np.sin(phi_coord)
    fading_line(ax_right, x_coord, y_coord, color)
    ax_right.plot(x_coord[0], y_coord[0], 'x', color='black', markersize=6)

for ax in [ax_left, ax_right]:
    ax.add_patch(plt.Circle((0, 0), 1, color='black'))
    ax.set_aspect('equal')
    ax.set_xlim(-6.5, 6.5)
    ax.set_ylim(-6.5, 6.5)
    ax.grid(True)

ax_left.set_title("Proper-Time Spatial Trajectories")
ax_right.set_title("Coordinate-Time Spatial Trajectories")
ax_left.set_xlabel("x")
ax_right.set_xlabel("x")
ax_left.set_ylabel("y")
ax_right.set_ylabel("y")

handles, labels = ax_left.get_legend_handles_labels()
fig_spatial.suptitle("Comparison of Spatial Trajectories", fontsize=14, y=0.98)
fig_spatial.legend(handles, labels, loc='lower center', ncol=len(Hs), fontsize=9, bbox_to_anchor=(0.5, -0.05))
plt.tight_layout(rect=[0, 0.05, 1, 0.95])
fig_spatial.savefig(os.path.join(output_dir, 'Q5_Spatial_Comparison_SideBySide_Faded.pdf'), dpi=300, bbox_inches='tight')
plt.show()
