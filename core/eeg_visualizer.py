import matplotlib.pyplot as plt
import numpy as np
import mne
from nilearn import surface
from neuromaps.datasets import fetch_fslr
from scipy.spatial.distance import cdist
import io
from PIL import Image, ImageEnhance
from matplotlib.colors import LinearSegmentedColormap

class EEGVisualizer:
    """
    Clinical Dashboard: Dynamic Spectrum & Global Topomaps.
    Synchronized with clinical event timeline.
    """
    def __init__(self, channel_names, motor_indices=None):
        self.channel_names = channel_names
        self.n_channels = len(channel_names)
        self.motor_indices = motor_indices if motor_indices else [0, 1, 2]
        self.smoothing_buffer = []
        self.beta_history = []
        self.buffer_size = 5
        self.stats = None
        
        # SOTA Editorial Theme (Warm bone/sand paper background)
        self.bg_color = '#F9F8F6'     # Warm bone/sand background
        self.text_color = '#1E293B'   # Deep slate-gray text
        self.border_color = '#94A3B8' # Slate border
        self.grid_color = '#EAE6E1'   # Sand gridline
        
        self.color_beta = '#E11D48'   # Crimson Rose (Pathology)
        self.color_delta = '#4F46E5'  # Royal Indigo (Intent)
        
        # Topomap Calibration
        fslr = fetch_fslr()
        self.nodes = np.vstack([surface.load_surf_mesh(fslr['midthickness'][0])[0][::15], 
                                surface.load_surf_mesh(fslr['midthickness'][1])[0][::15]])
        self.pos_2d = self.nodes[:, :2] 
        self.pos_2d -= np.mean(self.pos_2d, axis=0)
        self.pos_2d /= np.max(np.abs(self.pos_2d))
        
        # Extract actual standard 10-20 layout coordinates using MNE, handling non-standard/EOG channels
        montage = mne.channels.make_standard_montage('standard_1020')
        montage_ch_names = [ch.lower() for ch in montage.ch_names]
        
        valid_ch_names = [ch for ch in self.channel_names if ch.lower() in montage_ch_names]
        
        if valid_ch_names:
            sub_info = mne.create_info(ch_names=valid_ch_names, sfreq=250.0, ch_types='eeg')
            sub_info.set_montage(montage, on_missing='ignore')
            sub_layout = mne.channels.find_layout(sub_info)
            
            # Normalize layout positions
            sub_pos = sub_layout.pos[:, :2].copy()
            sub_pos -= np.mean(sub_pos, axis=0)
            sub_pos /= (np.max(np.abs(sub_pos)) + 1e-8)
            
            # Map back to full channel list
            self.sensor_grid = np.zeros((self.n_channels, 2))
            valid_idx = 0
            for i, ch in enumerate(self.channel_names):
                if ch in valid_ch_names:
                    self.sensor_grid[i] = sub_pos[valid_idx]
                    valid_idx += 1
                else:
                    # Place non-standard channels on an outer ring to prevent interference
                    angle = i * (2 * np.pi / self.n_channels)
                    self.sensor_grid[i] = [3.0 * np.cos(angle), 3.0 * np.sin(angle)]
        else:
            # Circular fallback
            self.sensor_grid = np.zeros((self.n_channels, 2))
            for i in range(self.n_channels):
                angle = i * (2 * np.pi / self.n_channels)
                self.sensor_grid[i] = [np.cos(angle), np.sin(angle)]
        
        dist_mat = cdist(self.pos_2d, self.sensor_grid)
        self.weights = np.exp(-dist_mat**2 / (2 * 0.20**2)) 
        self.weights /= (np.sum(self.weights, axis=1)[:, np.newaxis] + 1e-8)
        
        # Consistent Premium Divergent Colormap matching Timeline
        self.cmap = LinearSegmentedColormap.from_list('editorial_cmap', [
            (0.0, '#1E1B4B'),   # Deepest indigo-black (suppression)
            (0.20, '#4F46E5'),  # Royal indigo
            (0.28, '#E7E5E4'),  # Stone/neutral baseline (start)
            (0.30, '#E7E5E4'),  # Stone/neutral baseline (center)
            (0.32, '#E7E5E4'),  # Stone/neutral baseline (end)
            (0.40, '#E11D48'),  # Vibrant crimson rose
            (1.0, '#881337')    # Deep burgundy (severe pathology)
        ])
        
        # Figure caching for high-speed in-memory rendering
        self.fig_combined = None
        self.im_ax0 = None
        self.scatter_core = None
        self.scatter_halo = None
        self.title_text = None
        
        self.fig_brain = None
        self.scatter_brain_core = None
        self.scatter_brain_halo = None

    def reset_cache(self):
        """Clears cached figures to allow fresh rendering for a new subject/session."""
        if self.fig_combined is not None:
            plt.close(self.fig_combined)
            self.fig_combined = None
            self.im_ax0 = None
            self.scatter_core = None
            self.scatter_halo = None
            self.title_text = None
        if self.fig_brain is not None:
            plt.close(self.fig_brain)
            self.fig_brain = None
            self.scatter_brain_core = None
            self.scatter_brain_halo = None

    def render_combined_frame(self, eeg_features, psd, stats, target_dist, title, frame_idx, sim_frame, event_type=None):
        """Renders the unified clinical dashboard (Simulation + Smoothed Neural Graph) using in-memory caching."""
        # Use robust Z-scores for visualization to prevent freeze spikes from squashing walk dynamics
        median_all = stats.get('median', stats['mean'])
        mad_all = stats.get('mad', stats['std'])
        z_all = (eeg_features - median_all) / mad_all
        z_beta = z_all[:self.n_channels]
        
        # Broadband activation for the connectivity graph
        broadband_act = (z_all[:self.n_channels] + z_all[self.n_channels:]) / 2.0
        node_act = np.dot(self.weights, z_beta) 
        
        # Standardized square-root scaling for stable, smooth transition across frames/subjects
        node_act_scaled = np.sign(node_act) * np.power(np.abs(node_act), 0.5)
        
        vmin_dynamic = -1.5
        vmax_dynamic = 3.5
        
        if self.fig_combined is None:
            plt.ioff()
            self.fig_combined = plt.figure(figsize=(17, 9), facecolor=self.bg_color)
            plt.rcParams['text.color'] = self.text_color
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
            
            # --- AX 0: MAIN SIMULATION ---
            self.ax0 = self.fig_combined.add_axes([0.02, 0.02, 0.60, 0.96])
            self.im_ax0 = self.ax0.imshow(sim_frame)
            self.ax0.axis('off')
    
            # --- AX 1: MASSIVE NEURAL CONNECTIVITY NETWORK ---
            self.ax1 = self.fig_combined.add_axes([0.62, 0.02, 0.36, 0.96])
            self.ax1.set_facecolor(self.bg_color)
            
            # Title
            self.title_text = self.ax1.text(0.0, 1.25, title, fontsize=24, fontweight='bold', ha='center', va='center', color=self.text_color)
            
            # Neural Field (Halo Effect)
            self.scatter_halo = self.ax1.scatter(self.pos_2d[:, 0], self.pos_2d[:, 1], s=600, c=node_act_scaled, cmap=self.cmap, vmin=vmin_dynamic, vmax=vmax_dynamic, zorder=2, alpha=0.15)
            self.scatter_core = self.ax1.scatter(self.pos_2d[:, 0], self.pos_2d[:, 1], s=350, c=node_act_scaled, cmap=self.cmap, vmin=vmin_dynamic, vmax=vmax_dynamic, zorder=3, edgecolors='#475569', linewidths=0.8, alpha=1.0)
            self.ax1.set_aspect('equal')
            self.ax1.axis('off')
        else:
            # In-place updates for ultra-high-speed rendering
            self.im_ax0.set_data(sim_frame)
            self.title_text.set_text(title)
            self.scatter_halo.set_array(node_act_scaled)
            self.scatter_core.set_array(node_act_scaled)
            
        self.fig_combined.canvas.draw()
        rgba_buffer = self.fig_combined.canvas.buffer_rgba()
        width, height = self.fig_combined.canvas.get_width_height()
        img = Image.frombuffer("RGBA", (width, height), rgba_buffer, "raw", "RGBA", 0, 1).convert('RGB')
        return ImageEnhance.Sharpness(img).enhance(1.5)

    def render_combined_frame_fast(self, node_act_scaled, title, sim_frame):
        """Ultra-fast rendering using pre-calculated node_act_scaled to guarantee perfect sync."""
        vmin_dynamic = -1.5
        vmax_dynamic = 3.5
        
        if self.fig_combined is None:
            plt.ioff()
            self.fig_combined = plt.figure(figsize=(17, 9), facecolor=self.bg_color)
            plt.rcParams['text.color'] = self.text_color
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
            
            # --- AX 0: MAIN SIMULATION ---
            self.ax0 = self.fig_combined.add_axes([0.02, 0.02, 0.60, 0.96])
            self.im_ax0 = self.ax0.imshow(sim_frame)
            self.ax0.axis('off')
    
            # --- AX 1: MASSIVE NEURAL CONNECTIVITY NETWORK ---
            self.ax1 = self.fig_combined.add_axes([0.62, 0.02, 0.36, 0.96])
            self.ax1.set_facecolor(self.bg_color)
            
            # Title
            self.title_text = self.ax1.text(0.0, 1.25, title, fontsize=24, fontweight='bold', ha='center', va='center', color=self.text_color)
            
            # Neural Field (Halo Effect)
            self.scatter_halo = self.ax1.scatter(self.pos_2d[:, 0], self.pos_2d[:, 1], s=600, c=node_act_scaled, cmap=self.cmap, vmin=vmin_dynamic, vmax=vmax_dynamic, zorder=2, alpha=0.15)
            self.scatter_core = self.ax1.scatter(self.pos_2d[:, 0], self.pos_2d[:, 1], s=350, c=node_act_scaled, cmap=self.cmap, vmin=vmin_dynamic, vmax=vmax_dynamic, zorder=3, edgecolors='#475569', linewidths=0.8, alpha=1.0)
            self.ax1.set_aspect('equal')
            self.ax1.axis('off')
        else:
            # In-place updates
            self.im_ax0.set_data(sim_frame)
            self.title_text.set_text(title)
            self.scatter_halo.set_array(node_act_scaled)
            self.scatter_core.set_array(node_act_scaled)
            
        self.fig_combined.canvas.draw()
        rgba_buffer = self.fig_combined.canvas.buffer_rgba()
        width, height = self.fig_combined.canvas.get_width_height()
        img = Image.frombuffer("RGBA", (width, height), rgba_buffer, "raw", "RGBA", 0, 1).convert('RGB')
        return ImageEnhance.Sharpness(img).enhance(1.5)
    
    def render_brain_frame(self, eeg_features, stats, event_type=None):
        """Renders a standalone neural connectivity network with temporal smoothing using in-memory caching."""
        # Use robust Z-scores for visualization to prevent freeze spikes from squashing walk dynamics
        median_all = stats.get('median', stats['mean'])
        mad_all = stats.get('mad', stats['std'])
        z_all = (eeg_features - median_all) / mad_all
        z_beta = z_all[:self.n_channels]
        
        # Broadband activation for the connectivity graph
        broadband_act = (z_all[:self.n_channels] + z_all[self.n_channels:]) / 2.0
        node_act = np.dot(self.weights, z_beta) 
        
        # Standardized square-root scaling for stable, smooth transition across frames/subjects
        node_act_scaled = np.sign(node_act) * np.power(np.abs(node_act), 0.5)
        
        vmin_dynamic = -1.5
        vmax_dynamic = 3.5
        
        if self.fig_brain is None:
            plt.ioff()
            self.fig_brain = plt.figure(figsize=(9, 9), facecolor=self.bg_color)
            self.ax_brain = self.fig_brain.add_axes([0.05, 0.05, 0.9, 0.9])
            self.ax_brain.set_facecolor(self.bg_color)
            
            # Neural Field (Halo Effect)
            self.scatter_brain_halo = self.ax_brain.scatter(self.pos_2d[:, 0], self.pos_2d[:, 1], s=600, c=node_act_scaled, cmap=self.cmap, vmin=vmin_dynamic, vmax=vmax_dynamic, zorder=2, alpha=0.15)
            self.scatter_brain_core = self.ax_brain.scatter(self.pos_2d[:, 0], self.pos_2d[:, 1], s=350, c=node_act_scaled, cmap=self.cmap, vmin=vmin_dynamic, vmax=vmax_dynamic, zorder=3, edgecolors='#475569', linewidths=0.8, alpha=1.0)
            self.ax_brain.set_aspect('equal')
            self.ax_brain.axis('off')
        else:
            # In-place updates for ultra-high-speed rendering
            self.scatter_brain_halo.set_array(node_act_scaled)
            self.scatter_brain_core.set_array(node_act_scaled)
            
        self.fig_brain.canvas.draw()
        rgba_buffer = self.fig_brain.canvas.buffer_rgba()
        width, height = self.fig_brain.canvas.get_width_height()
        return Image.frombuffer("RGBA", (width, height), rgba_buffer, "raw", "RGBA", 0, 1).convert('RGB')

    def save_gif(self, frames, filename='output.gif', duration=100):
        print(f"Exporting Dashboard Animation: {filename}...")
        frames[0].save(filename, save_all=True, append_images=frames[1:], duration=duration, loop=0, optimize=False)

def render_eeg_worker_ext(args):
    """Pickleable top-level helper function for parallel pool rendering."""
    viz, eeg_features, osc_state, title, frame_idx, sim_frame, progress, event = args
    stats = getattr(viz, 'stats', None)
    if stats is None:
        stats = {'mean': np.zeros_like(eeg_features), 'std': np.ones_like(eeg_features)}
    return viz.render_combined_frame(
        eeg_features, osc_state, stats, progress, title, frame_idx, sim_frame, event
    )
