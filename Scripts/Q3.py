import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
import os

# Directory for saving figures
output_dir = r"C:\CATAM_BlackHoleOrbits\Figures_BH"
os.makedirs(output_dir, exist_ok=True)

# Parameters
Hs = [0.0, 0.5, 1.0, 1.5, 2.0]
epsilon = 1
k = np.sqrt(25/27)
ds = 0.01
max_steps = int(200*np.pi/ds)
cycle_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Radii: old (r0=6 for capture points), new (r0=2.5)
r0_old = 6.0
r0 = 2.5

def F(r):
    return 1 - 1/r

def dVeff_dr(r,h):
    return -2*h**2/r**3 + 1/r**2 + 3*h**2/r**4

def derivatives(y,h):
    r, r_dot, phi, s = y
    return np.array([r_dot, -0.5*dVeff_dr(r,h), h/r**2, 1.0])

# Integrator
def integrate_orbit(r0_val, h):
    y = np.array([r0_val, 0.0, 0.0, 0.0])
    traj = []
    steps = int(np.ceil((2*np.pi)/(h/r0_val**2)/ds)) if h==2.0 else max_steps
    for _ in range(steps):
        traj.append(y.copy())
        r = y[0]
        if h!=2.0 and r<=1.0:
            break
        k1 = derivatives(y,h)
        k2 = derivatives(y+0.5*ds*k1, h)
        k3 = derivatives(y+0.5*ds*k2, h)
        k4 = derivatives(y+ds*k3, h)
        y += (ds/6)*(k1 + 2*k2 + 2*k3 + k4)
    return np.array(traj)

# Compute capture angles at r0_old
capture_phis_old = {}
captured_old = []
for h in Hs:
    traj_old = integrate_orbit(r0_old, h)
    if traj_old[-1,0] <= 1.0 and h!=2.0:
        captured_old.append(h)
        capture_phis_old[h] = traj_old[-1,2]

# Simulate for r0=2.5
captured = []
capture_times = {}

fig1, ax1 = plt.subplots(figsize=(6,6))
handles = []
for idx,h in enumerate(Hs):
    traj = integrate_orbit(r0, h)
    if traj[-1,0] <= 1.0:
        captured.append(h)
        capture_times[h] = traj[-1,3]
    r_vals,phi_vals = traj[:,0],traj[:,2]
    x = r_vals*np.cos(phi_vals)
    y = r_vals*np.sin(phi_vals)
    rgba = mcolors.to_rgba(cycle_colors[idx])
    alphas = np.linspace(1,0.01,len(x))
    colors = [(rgba[0],rgba[1],rgba[2],a) for a in alphas]
    ax1.scatter(x,y,color=colors,s=2)
    handles.append(Line2D([0],[0],color=cycle_colors[idx],label=f"h = {h}"))
# Start & BH markers
h0,=ax1.plot(r0,0,'x',color='black',markersize=10,linestyle='None',label='Start')
handles.append(h0)
bh=plt.Circle((0,0),1,color='black'); ax1.add_patch(bh)
h1=Line2D([0],[0],marker='o',color='black',linestyle='None',markersize=10,label='Black Hole')
handles.append(h1)
ax1.legend(handles=handles,loc='upper right')
ax1.set(aspect='equal',xlim=[-6.5,6.5],ylim=[-6.5,6.5],xlabel='x',ylabel='y',
         title=r'Orbits around a Black Hole with a Closer Start Point')
ax1.grid(True)
plt.savefig(os.path.join(output_dir,'Q3_Orbits_xy_r25.pdf'),dpi=300,bbox_inches='tight')

# r-phi plot with both old and new capture lines
fig2, ax2 = plt.subplots(figsize=(6,4))
ax2.axhspan(0,1,color='darkgrey',alpha=0.6)
for idx,h in enumerate(Hs):
    traj = integrate_orbit(r0,h)
    ax2.plot(traj[:,2],traj[:,0],color=cycle_colors[idx],label=f"h = {h}")
# old capture lines (dashed, light)
for h in captured_old:
    phi_old = capture_phis_old[h]
    ax2.axvline(phi_old,color='darkgrey',linestyle='--',alpha=0.6)
# new capture lines (dashed, color)
for h in captured:
    phi_new = capture_times[h] and integrate_orbit(r0,h)[-1,2]
    ax2.axvline(phi_new,color=cycle_colors[Hs.index(h)],linestyle='--')
ax2.axhline(1,color='black',linestyle='--')
ax2.set_ylim(bottom=0)
ax2.set(xlabel=r'$\phi$',ylabel='r',title='Radius of Orbit as a Function of Angle with a Closer Start Point')
ax2.legend(loc='upper right')
ax2.grid(True)
plt.savefig(os.path.join(output_dir,'Q3_Orbits_rphi_r25.pdf'),dpi=300,bbox_inches='tight')
plt.show()
