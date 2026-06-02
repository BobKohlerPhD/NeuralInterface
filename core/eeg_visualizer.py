import matplotlib.pyplot as plt
import numpy as np
from nilearn import surface
from neuromaps.datasets import fetch_fslr
from scipy.spatial.distance import cdist
import io
from PIL import Image, ImageEnhance

class EEGVisualizer:
    """
    Clinical Dashboard: Dynamic Spectrum & Global Topomaps.
    Synchronized with Neural Brake biomarkers for Subject 038.
    """
    def __init__(self, channel_names, motor_indices=None):
        self.channel_names = channel_names
        self.n_channels = len(channel_names)
        self.motor_indices = motor_indices if motor_indices else [0, 1, 2]
        self.smoothing_buffer = []
        self.beta_history = []
        self.buffer_size = 5
        
        # Dashboard Theme
        self.bg_color = '#000000'
        self.cyan = '#00f2ff' 
        self.red = '#ff1100'   
        
        # Topomap Calibration
        fslr = fetch_fslr()
        self.nodes = np.vstack([surface.load_surf_mesh(fslr['midthickness'][0])[0][::15], 
                                surface.load_surf_mesh(fslr['midthickness'][1])[0][::15]])
        self.pos_2d = self.nodes[:, :2] 
        self.pos_2d -= np.mean(self.pos_2d, axis=0)
        self.pos_2d /= np.max(np.abs(self.pos_2d))
        
        grid_dim = int(np.ceil(np.sqrt(self.n_channels)))
        x, y = np.meshgrid(np.linspace(-0.85, 0.85, grid_dim), np.linspace(-0.85, 0.85, grid_dim))
        self.sensor_grid = np.column_stack([x.ravel(), y.ravel()])[:self.n_channels]
        
        dist_mat = cdist(self.pos_2d, self.sensor_grid)
        self.weights = np.exp(-dist_mat**2 / (2 * 0.12**2)) 
        self.weights /= (np.sum(self.weights, axis=1)[:, np.newaxis] + 1e-8)
        self.cmap = plt.cm.magma

    def render_combined_frame(self, eeg_features, psd, stats, target_dist, title, frame_idx, sim_frame, event_type=None):
        """Renders the unified clinical dashboard (Simulation + Smoothed Neural Graph)."""
        # Normalize
        z_all = (eeg_features - stats['mean']) / stats['std']
        z_beta = z_all[:self.n_channels]
        
        # Broadband activation for the connectivity graph
        broadband_act = (z_all[:self.n_channels] + z_all[self.n_channels:]) / 2.0
        node_act = np.dot(self.weights, z_beta) 
        
        # Temporal Smoothing: EMA (Unifies combined and standalone views)
        local_vmax = np.percentile(node_act, 99.9)
        local_vmin = np.percentile(node_act, 0.1)
        
        if not hasattr(self, 'running_vmax_comb'):
            self.running_vmax_comb, self.running_vmin_comb = local_vmax, local_vmin
        else:
            alpha = 0.15 
            self.running_vmax_comb = alpha * local_vmax + (1 - alpha) * self.running_vmax_comb
            self.running_vmin_comb = alpha * local_vmin + (1 - alpha) * self.running_vmin_comb

        vmax_dynamic = max(1e-6, self.running_vmax_comb)
        vmin_dynamic = self.running_vmin_comb
        
        fig = plt.figure(figsize=(34, 18), facecolor=self.bg_color)
        
        # --- AX 0: MAIN SIMULATION ---
        ax0 = fig.add_axes([0.02, 0.02, 0.60, 0.96])
        ax0.imshow(sim_frame); ax0.axis('off')

        # --- AX 1: MASSIVE NEURAL CONNECTIVITY NETWORK ---
        ax1 = fig.add_axes([0.62, 0.02, 0.36, 0.96])
        
        # Neural Field (Halo Effect)
        ax1.scatter(self.pos_2d[:, 0], self.pos_2d[:, 1], s=350, c=node_act, cmap=self.cmap, vmin=vmin_dynamic, vmax=vmax_dynamic, zorder=3, edgecolors='white', linewidths=1.5, alpha=1.0)
        ax1.scatter(self.pos_2d[:, 0], self.pos_2d[:, 1], s=600, c=node_act, cmap=self.cmap, vmin=vmin_dynamic, vmax=vmax_dynamic, zorder=2, alpha=0.3)
        
        # Broadband Connectivity (Electric Graph)
        motor_pos = self.sensor_grid[self.motor_indices]
        local_act = broadband_act[self.motor_indices]
        local_min, local_max = local_act.min(), local_act.max()
        range_val = (local_max - local_min) + 1e-6
        
        for i in range(len(motor_pos)):
            for j in range(i + 1, len(motor_pos)):
                s_i = (broadband_act[self.motor_indices[i]] - local_min) / range_val
                s_j = (broadband_act[self.motor_indices[j]] - local_min) / range_val
                strength = (s_i + s_j) / 2.0
                
                if strength > 0.1: 
                    ax1.plot([motor_pos[i,0], motor_pos[j,0]], [motor_pos[i,1], motor_pos[j,1]], 
                             color='#00ffff', alpha=strength * 0.75, linewidth=1 + 12 * strength, zorder=1,
                             solid_capstyle='round')

        # State Indicator
        is_freezing = (event_type == 'break cnt')
        halo_color = '#ff0000' if is_freezing else 'white'
        halo_alpha = 0.5 if is_freezing else 0.2
        ax1.add_patch(plt.Circle((0, 0), 1.1, color=halo_color, fill=False, linewidth=10 if is_freezing else 5, alpha=halo_alpha))
        
        ax1.set_aspect('equal'); ax1.axis('off')

        buf = io.BytesIO()
        plt.savefig(buf, format='png', facecolor=self.bg_color)
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        return ImageEnhance.Sharpness(img).enhance(1.5)

    def render_brain_frame(self, eeg_features, stats, event_type=None):
        """Renders a standalone neural connectivity network with temporal smoothing."""
        # Normalize
        z_all = (eeg_features - stats['mean']) / stats['std']
        z_beta = z_all[:self.n_channels]
        
        # Broadband activation for the connectivity graph
        broadband_act = (z_all[:self.n_channels] + z_all[self.n_channels:]) / 2.0
        node_act = np.dot(self.weights, z_beta) 
        
        # Temporal Smoothing: Exponential Moving Average (EMA)
        local_vmax = np.percentile(node_act, 99.9)
        local_vmin = np.percentile(node_act, 0.1)
        
        if not hasattr(self, 'running_vmax'):
            self.running_vmax, self.running_vmin = local_vmax, local_vmin
        else:
            alpha = 0.15 
            self.running_vmax = alpha * local_vmax + (1 - alpha) * self.running_vmax
            self.running_vmin = alpha * local_vmin + (1 - alpha) * self.running_vmin

        vmax_dynamic = max(1e-6, self.running_vmax)
        vmin_dynamic = self.running_vmin
        
        fig = plt.figure(figsize=(18, 18), facecolor=self.bg_color)
        ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
        
        # Neural Field (Halo Effect)
        ax.scatter(self.pos_2d[:, 0], self.pos_2d[:, 1], s=350, c=node_act, cmap=self.cmap, vmin=vmin_dynamic, vmax=vmax_dynamic, zorder=3, edgecolors='white', linewidths=1.5, alpha=1.0)
        ax.scatter(self.pos_2d[:, 0], self.pos_2d[:, 1], s=600, c=node_act, cmap=self.cmap, vmin=vmin_dynamic, vmax=vmax_dynamic, zorder=2, alpha=0.3)
        
        # Broadband Connectivity (Always-On Electric Graph)
        motor_pos = self.sensor_grid[self.motor_indices]
        local_act = broadband_act[self.motor_indices]
        local_min, local_max = local_act.min(), local_act.max()
        range_val = (local_max - local_min) + 1e-6
        
        for i in range(len(motor_pos)):
            for j in range(i + 1, len(motor_pos)):
                s_i = (broadband_act[self.motor_indices[i]] - local_min) / range_val
                s_j = (broadband_act[self.motor_indices[j]] - local_min) / range_val
                strength = (s_i + s_j) / 2.0
                
                if strength > 0.1: 
                    ax.plot([motor_pos[i,0], motor_pos[j,0]], [motor_pos[i,1], motor_pos[j,1]], 
                             color='#00ffff', alpha=strength * 0.75, linewidth=1 + 12 * strength, zorder=1,
                             solid_capstyle='round')

        # State indicator: Turn Red during Freezing of Gait
        is_freezing = (event_type == 'break cnt')
        halo_color = '#ff0000' if is_freezing else 'white'
        halo_alpha = 0.5 if is_freezing else 0.2
        ax.add_patch(plt.Circle((0, 0), 1.1, color=halo_color, fill=False, linewidth=10 if is_freezing else 5, alpha=halo_alpha))
        
        ax.set_aspect('equal'); ax.axis('off')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', facecolor=self.bg_color)
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).convert('RGB')

    def save_gif(self, frames, filename='output.gif', duration=100):
        print(f"Exporting Dashboard Animation: {filename}...")
        frames[0].save(filename, save_all=True, append_images=frames[1:], duration=duration, loop=0, optimize=True)
