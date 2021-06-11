import numpy as np
def generate_heatmap_target(heatmap_size, landmarks, sigmas, scale=1.0, normalize=False):
    x_dim, y_dim = heatmap_size
    heatmap = np.zeros((x_dim, y_dim), dtype=float)
    