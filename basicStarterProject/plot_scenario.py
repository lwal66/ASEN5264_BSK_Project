"""
plot_scenario.py  -  Ground track and city target visualization.

Runs one greedy PPO episode, records the satellite ECI position at each
step, converts to geodetic lat/lon, and plots:
  (a) Ground track coloured by simulation time
  (b) All 1000 city targets coloured by priority
  (c) Successfully imaged cities highlighted with stars
  (d) Battery fraction over episode progress

Uses cartopy if available, otherwise falls back to a plain matplotlib map.

Usage
-----
    python plot_scenario.py
    python plot_scenario.py --max-steps 200
    python plot_scenario.py --no-ppo      # heuristic policy instead
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch

from Basilisk.architecture import bskLogging

from ppo import ActorCritic, ACTION_NAMES
from ppo_eval import find_latest_checkpoint, load_checkpoint, heuristic_action
from config import EnvConfig, TrainConfig
from envs import make_env

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

# Journal style
mpl.rcParams.update({
    "font.size":         8,
    "axes.labelsize":    8,
    "xtick.labelsize":   7,
    "ytick.labelsize":   7,
    "legend.fontsize":   7,
    "lines.linewidth":   1.2,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.02,
})


# ---------------------------------------------------------------------------
# ECI → lat/lon
# ---------------------------------------------------------------------------

def eci_to_latlon(r_ECI: np.ndarray, t_seconds: float):
    x, y, z = r_ECI
    r      = np.linalg.norm(r_ECI)
    lat    = np.degrees(np.arcsin(np.clip(z / r, -1, 1)))
    gmst   = (t_seconds / 86164.1) * 360.0 % 360.0
    ra     = np.degrees(np.arctan2(y, x))
    lon    = (ra - gmst + 180) % 360 - 180
    return lat, lon


# ---------------------------------------------------------------------------
# Rollout with ground track recording
# ---------------------------------------------------------------------------

def run_episode(model, max_steps, use_heuristic=False):
    bskLogging.setDefaultLogLevel(bskLogging.BSK_WARNING)
    env_cfg = EnvConfig()
    env     = make_env(env_cfg)
    obs_np, _ = env.reset(seed=env_cfg.seed)

    # Unwrap to access BSK state
    raw_env = env
    while hasattr(raw_env, "env"):
        raw_env = raw_env.env

    track   = []   # (lat, lon, time, battery, action)
    imaged  = []   # (lat, lon, priority, city_name)
    done    = False
    step    = 0

    while not done and step < max_steps:
        if use_heuristic:
            action = heuristic_action(obs_np)
        else:
            obs_t = torch.as_tensor(obs_np, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                dist, _ = model(obs_t)
                action   = dist.probs.argmax(dim=-1).item()

        try:
            r_ECI    = np.array(raw_env.satellite.dynamics.r_BN_N)
            sim_time = float(raw_env.satellite.simulator.sim_time)
            lat, lon = eci_to_latlon(r_ECI, sim_time)
        except Exception:
            lat, lon, sim_time = 0.0, 0.0, 0.0

        try:
            next_obs, reward, terminated, truncated, _ = env.step(action)
        except RuntimeError:
            break

        battery = float(next_obs[0])
        track.append((lat, lon, sim_time, battery, action))

        if reward > 0:
            try:
                tgt = getattr(raw_env.satellite, "latest_target", None)
                if tgt is not None:
                    tgt_lat = float(np.degrees(tgt.latitude))
                    tgt_lon = float(np.degrees(tgt.longitude))
                    priority = float(getattr(tgt, "priority", 0.5))
                    imaged.append((tgt_lat, tgt_lon, priority, tgt.name))
            except Exception:
                pass

        obs_np = next_obs
        step  += 1
        done   = terminated or truncated

    # Get all city targets from scenario
    all_targets = []
    try:
        scenario = raw_env.world.target_list
        for tgt in scenario:
            tgt_lat  = float(np.degrees(tgt.latitude))
            tgt_lon  = float(np.degrees(tgt.longitude))
            priority = float(getattr(tgt, "priority", 0.5))
            all_targets.append((tgt_lat, tgt_lon, priority))
    except Exception:
        pass

    env.close()
    return track, imaged, all_targets


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_ground_track(track, imaged, all_targets, title="Simulation Scenario"):
    lats     = [r[0] for r in track]
    lons     = [r[1] for r in track]
    times    = [r[2] for r in track]
    batteries = [r[3] for r in track]
    norm_t   = np.linspace(0, 1, len(times))

    # ── Build map axes - cartopy if available, else plain matplotlib ─────────
    fig = plt.figure(figsize=(7, 3.5))
    fig.suptitle(title, fontsize=9, fontweight="bold")

    if HAS_CARTOPY:
        ax_map = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax_map.set_global()
        # NASA Blue Marble as background
        ax_map.stock_img()
        ax_map.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor="white", alpha=0.6)
        ax_map.add_feature(cfeature.BORDERS,   linewidth=0.2, linestyle=":",
                           edgecolor="white", alpha=0.4)
        ax_map.gridlines(draw_labels=True, linewidth=0.2, color="white",
                         alpha=0.3, x_inline=False, y_inline=False)
        transform = ccrs.PlateCarree()
    else:
        # Fallback: download NASA Blue Marble and display as imshow background
        import urllib.request, io
        from PIL import Image
        ax_map = fig.add_subplot(1, 1, 1)
        ax_map.set_xlim(-180, 180)
        ax_map.set_ylim(-90, 90)
        ax_map.set_xlabel("Longitude (°)")
        ax_map.set_ylabel("Latitude (°)")

        # Try to load a cached earth image or download it
        earth_cache = Path(__file__).parent / "earth_nasa.jpg"
        if not earth_cache.exists():
            print("  Downloading NASA Blue Marble image (one-time)...")
            url = "https://eoimages.gsfc.nasa.gov/images/imagerecords/74000/74167/world.200412.3x5400x2700.jpg"
            try:
                urllib.request.urlretrieve(url, earth_cache)
            except Exception:
                earth_cache = None

        if earth_cache and earth_cache.exists():
            img = np.array(Image.open(earth_cache))
            ax_map.imshow(img, extent=[-180, 180, -90, 90], aspect="auto",
                          zorder=0, origin="upper")
        else:
            ax_map.set_facecolor("#0a1628")

        ax_map.grid(True, linewidth=0.2, color="white", alpha=0.2)
        transform = None

    def _scatter(ax, x, y, **kwargs):
        if HAS_CARTOPY:
            return ax.scatter(x, y, transform=transform, **kwargs)
        return ax.scatter(x, y, **kwargs)

    def _plot(ax, x, y, **kwargs):
        if HAS_CARTOPY:
            return ax.plot(x, y, transform=transform, **kwargs)
        return ax.plot(x, y, **kwargs)

    # All city targets - coloured by priority
    if all_targets:
        t_lats = [t[0] for t in all_targets]
        t_lons = [t[1] for t in all_targets]
        t_pris = [t[2] for t in all_targets]
        sc_all = _scatter(ax_map, t_lons, t_lats, c=t_pris,
                          cmap="RdYlGn", vmin=0, vmax=1,
                          s=4, alpha=0.7, zorder=2, label="City targets")
        cbar = plt.colorbar(sc_all, ax=ax_map, orientation="vertical",
                            shrink=0.6, pad=0.02)
        cbar.set_label("Priority", fontsize=7)
        cbar.ax.tick_params(labelsize=6)

    # Ground track coloured by time
    _scatter(ax_map, lons, lats, c=norm_t, cmap="plasma",
             s=8, zorder=3, alpha=0.9)
    _plot(ax_map, lons, lats, color="white", linewidth=0.3, alpha=0.3)

    # Imaged cities - gold stars
    if imaged:
        i_lats = [i[0] for i in imaged]
        i_lons = [i[1] for i in imaged]
        _scatter(ax_map, i_lons, i_lats, c="gold", s=50, marker="*",
                 edgecolors="black", linewidths=0.3, zorder=5,
                 label=f"Imaged ({len(imaged)})")

    # Start / end markers
    _scatter(ax_map, [lons[0]],  [lats[0]],  c="lime", s=70, marker="^",
             edgecolors="black", linewidths=0.4, zorder=6, label="Start")
    _scatter(ax_map, [lons[-1]], [lats[-1]], c="red",  s=70, marker="v",
             edgecolors="black", linewidths=0.4, zorder=6, label="End")

    ax_map.legend(loc="lower left", fontsize=6, markerscale=1.2,
                  framealpha=0.7, facecolor="#222222", labelcolor="white",
                  edgecolor="gray")
    ax_map.set_title("(a) Satellite Ground Track", fontsize=8, loc="left")

    plt.tight_layout(pad=0.5)
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot simulation ground track")
    parser.add_argument("--max-steps", type=int, default=100,
                        help="Max steps per episode")
    parser.add_argument("--no-ppo",    action="store_true",
                        help="Use heuristic policy instead of trained PPO")
    args = parser.parse_args()

    bskLogging.setDefaultLogLevel(bskLogging.BSK_WARNING)
    paths = TrainConfig()
    paths.eval_dir.mkdir(parents=True, exist_ok=True)

    if args.no_ppo:
        model = None
        policy_label = "Heuristic"
        print("Using heuristic policy...")
    else:
        checkpoint_path = find_latest_checkpoint(paths.checkpoint_dir)
        model, _, _, _ = load_checkpoint(checkpoint_path)
        model.eval()
        policy_label = checkpoint_path.stem
        print(f"Using checkpoint: {checkpoint_path.name}")

    print(f"Running episode (max {args.max_steps} steps)...")
    track, imaged, all_targets = run_episode(
        model, args.max_steps, use_heuristic=args.no_ppo
    )

    print(f"  Steps: {len(track)}")
    print(f"  Cities imaged: {len(imaged)}")
    if not HAS_CARTOPY:
        print("  Note: cartopy not installed - using basic matplotlib map.")
        print("  Install with: pip install cartopy")

    fig = plot_ground_track(
        track, imaged, all_targets,
        title=f"BSK-RL Simulation - {policy_label}"
    )

    out_path = paths.eval_dir / "scenario_ground_track.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()