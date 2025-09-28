import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
import os

# Directory for saving figures
output_dir = r'C:\CATAM_BlackHoleOrbits\Figures_BH'
os.makedirs(output_dir, exist_ok=True)

# Parameters
Hs = np.linspace(0.0, 2.0, 200) 
r0_list = [6.0, 2.5]               # starting radii

# Constant energy
k_const = np.sqrt(25/27)

# Functions
def Veff(r, h):
    return (1 - 1/r) * (1 + h**2 / r**2)

def fall_time(h, r0):
    integrand = lambda rr: 1/mp.sqrt(k_const**2 - Veff(rr, h))
    try:
        τ = mp.quad(integrand, [1, r0])
        return float(mp.re(τ))
    except ValueError:
        return np.nan

# Compute proper fall-in times
tau_data = {r0: [] for r0 in r0_list}
for r0 in r0_list:
    h_cir = np.sqrt(r0**2 / (2*r0 - 3))
    cutoff = h_cir if r0 == 6.0 else 2.0
    for h in Hs:
        tau_data[r0].append(np.nan if h >= cutoff else fall_time(h, r0))

# Figure 1: Proper fall-in time
fig, ax = plt.subplots(figsize=(6,4))
colors = ['red', 'blue']
for r0, color in zip(r0_list, colors):
    ax.plot(Hs, tau_data[r0], color=color, label=f'$r_0={r0}$')

# Vertical lines at both circular thresholds
h_cir_60 = np.sqrt(6.0**2 / (2*6.0 - 3))      # = 2.0
h_cir_25 = np.sqrt(2.5**2 / (2*2.5 - 3))      # ≈1.7678
ax.axvline(h_cir_60, linestyle='--', color='black')
ax.axvline(h_cir_25, linestyle='--', color='black')

# Customize x-ticks
ticks = [t for t in ax.get_xticks() if abs(t - 1.75) > 1e-2]
ticks.extend([h_cir_25, h_cir_60])
ticks = sorted(set(ticks))
ax.set_xticks(ticks)

for label in ax.get_xticklabels():
    try:
        val = float(label.get_text())
        if abs(val - h_cir_25) < 1e-3 or abs(val - h_cir_60) < 1e-3:
            label.set_fontweight('bold')
    except ValueError:
        pass

ax.set_xlabel('$h$')
ax.set_ylabel('$\\tau$')
ax.set_title('Proper Fall-In Time')
ax.legend(loc='best')
ax.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Q4_Timelot_extended.pdf'), dpi=300, bbox_inches='tight')

# Figure 2: Effective potential stability
fig2, ax2 = plt.subplots(figsize=(6,4))
for r0, color in zip(r0_list, colors):
    h_cir = np.sqrt(r0**2 / (2*r0 - 3))
    r_vals = np.linspace(max(1.01, r0 - 1), r0 + 1, 400)
    ax2.plot(r_vals, Veff(r_vals, h_cir), color=color,
             label=fr'$r_0={r0},\ h_{{\rm cir}}={h_cir:.3f}$')
    ax2.axvline(r0, linestyle='--', color='black')

ax2.set_xlabel('$r$')
ax2.set_ylabel('$V_{\\mathrm{eff}}(r)$')
ax2.set_title('Effective Potential Stability')
ax2.legend(loc='best')
ax2.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Q4_Veff_stability.pdf'), dpi=300, bbox_inches='tight')

plt.show()
