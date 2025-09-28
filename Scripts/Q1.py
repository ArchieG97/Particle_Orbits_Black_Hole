import numpy as np
import matplotlib.pyplot as plt
import os

# Directory for saving figures
output_dir = r"C:\\CATAM_BlackHoleOrbits\\Figures_BH"
os.makedirs(output_dir, exist_ok=True)

epsilon = 1
h_values = [1, 2, 3, 4]

r = np.linspace(0.5, 10, 1000)

def V_eff(r, h):
    return (1 - 1/r) * (epsilon + h**2 / r**2)

plt.figure(figsize=(8, 5))
for h in h_values:
    plt.plot(r, V_eff(r, h), label=f'h = {h}')

plt.axvline(1, linestyle='--', color='gray')  # vertical dashed line at r=1
current_ticks = plt.xticks()[0].tolist()
if 1 not in current_ticks:
    current_ticks.append(1)

# Sort ticks
current_ticks = sorted(current_ticks)

# Create labels, marking 1 as 'r=1'
labels = [ 'r=1' if tick==1 else str(int(tick) if tick.is_integer() else f'{tick:.1f}') for tick in current_ticks ]
plt.xticks(current_ticks, labels)

plt.xlabel('r')
plt.ylabel(r'$V_{\mathrm{eff}}(r)$')
plt.title('Effective Potential for Various Angular Momenta')
plt.legend()
plt.grid()

output_path = os.path.join(output_dir, 'Q1_V_eff.pdf')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()
