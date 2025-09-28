import numpy as np
import matplotlib.pyplot as plt
import os

# Output directory
output_dir = r"C:\CATAM_BlackHoleOrbits\Figures_BH"
os.makedirs(output_dir, exist_ok=True)

def capture_cross_section(v: np.ndarray) -> np.ndarray:
    return (9 * np.pi / 4) * (1 + 2 * v**2) / v**2

def plot_sigma_subplots(v_min=0.01, v_max=0.999,
                        zoom_min=0.3, zoom_max=0.999, n_points=1000):

    # full range
    v_full     = np.linspace(v_min, v_max, n_points)
    sigma_full = capture_cross_section(v_full)

    # zoom range
    v_zoom     = np.linspace(zoom_min, zoom_max, n_points)
    sigma_zoom = capture_cross_section(v_zoom)

    sigma_photon_limit = 27 * np.pi / 4

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Full-range plot (black) with zoom segment in blue
    axes[0].plot(v_full, sigma_full, color='k', linewidth=2)
    mask = v_full >= zoom_min
    axes[0].plot(v_full[mask], sigma_full[mask], color='b', linewidth=2)
    axes[0].set_xlabel('v')
    axes[0].set_ylabel(r'$\sigma(v)$')
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # Zoomed-in plot in blue
    axes[1].plot(v_zoom, sigma_zoom, color='b', linewidth=2)
    axes[1].set_xlabel('v')
    axes[1].grid(True, linestyle='--', alpha=0.5)

    # Horizontal dashed line at the photon limit
    axes[1].axhline(y=sigma_photon_limit, color='k', linestyle='--')

    # Build yticks
    yticks = list(axes[1].get_yticks())
    yticks = [t for t in yticks if not np.isclose(t, 20.0)]
    yticks.append(sigma_photon_limit)
    axes[1].set_yticks(yticks)

    # Build matching tick labels
    yticklabels = [f"{y:.1f}" for y in yticks]
    yticklabels[-1] = r"$\frac{27\pi}{4}$"
    axes[1].set_yticklabels(yticklabels)

    # Set custom y-limits for the zoomed plot
    min_zoom = np.min(sigma_zoom)
    max_zoom = np.max(sigma_zoom)
    axes[1].set_ylim(min_zoom * 0.9, max_zoom * 1.1)

    # Main title—positioned
    fig.suptitle("Black Hole Capture Cross-Section", y=0.95, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.9])

    plt.savefig(os.path.join(output_dir, 'Q6_sigma_subplots.pdf'), dpi=300)
    plt.show()

if __name__ == '__main__':
    plot_sigma_subplots()
