# viz_depth_dataset.py
#
# OpenCV-based utility to browse and inspect the generated depth dataset.
#
# Controls:
#   →  / D      : next image
#   ←  / A      : previous image
#   SPACE       : toggle auto-play
#   +  / -      : speed up / slow down auto-play
#   T           : switch between train / test set
#   H           : toggle histogram panel
#   S           : save current image as PNG
#   R           : jump to random image
#   G           : enter index (type number + Enter)
#   ESC / Q     : quit
#
# Usage:
#   python viz_depth_dataset.py
#   python viz_depth_dataset.py --data_dir depth_vae_data --colormap inferno
#   python viz_depth_dataset.py --scale 10 --show_histogram
#
import argparse
import os
import glob
import numpy as np
import cv2


def parse_args():
    p = argparse.ArgumentParser(description="Browse depth dataset")
    p.add_argument("--data_dir",  default="depth_vae_data")
    p.add_argument("--split",     default="train", choices=["train", "test"])
    p.add_argument("--scale",     type=int, default=8, help="Display upscale factor")
    p.add_argument("--colormap",  default="viridis",
                   choices=["viridis", "inferno", "magma", "jet", "bone", "turbo"])
    p.add_argument("--show_histogram", action="store_true")
    p.add_argument("--max_depth", type=float, default=None,
                   help="Override max depth (auto-detected from metadata)")
    return p.parse_args()


COLORMAPS = {
    "viridis": cv2.COLORMAP_VIRIDIS,
    "inferno": cv2.COLORMAP_INFERNO,
    "magma":   cv2.COLORMAP_MAGMA,
    "jet":     cv2.COLORMAP_JET,
    "bone":    cv2.COLORMAP_BONE,
    "turbo":   cv2.COLORMAP_TURBO,
}


def load_file_list(data_dir, split):
    """Load sorted list of .npy files for given split."""
    split_dir = os.path.join(data_dir, split)
    files = sorted(glob.glob(os.path.join(split_dir, "*.npy")))
    return files


def depth_to_bgr(depth, max_depth, colormap_cv):
    """Convert depth (H, W) float32 → (H, W, 3) BGR uint8."""
    norm = np.clip(depth / max(max_depth, 1e-6), 0.0, 1.0)
    gray = (norm * 255).astype(np.uint8)
    return cv2.applyColorMap(gray, colormap_cv)


def make_histogram(depth, max_depth, hist_w=300, hist_h=150):
    """Create a histogram image of depth values."""
    canvas = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)
    canvas[:] = (30, 30, 30)

    valid = depth[depth < max_depth * 0.99]
    if len(valid) < 10:
        cv2.putText(canvas, "No valid pixels", (10, hist_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        return canvas

    n_bins = 50
    counts, edges = np.histogram(valid, bins=n_bins, range=(0, max_depth))
    max_count = max(counts.max(), 1)

    bar_w = hist_w // n_bins
    for i in range(n_bins):
        bar_h = int(counts[i] / max_count * (hist_h - 30))
        x0 = i * bar_w
        y0 = hist_h - 5 - bar_h
        color_val = int(i / n_bins * 255)
        color = cv2.applyColorMap(np.array([[color_val]], dtype=np.uint8),
                                   cv2.COLORMAP_VIRIDIS)[0, 0].tolist()
        cv2.rectangle(canvas, (x0, y0), (x0 + bar_w - 1, hist_h - 5), color, -1)

    # Axis labels
    cv2.putText(canvas, "0", (2, hist_h - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
    cv2.putText(canvas, f"{max_depth:.1f}m", (hist_w - 45, hist_h - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
    cv2.putText(canvas, "Depth histogram", (5, 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    return canvas


def make_stats_overlay(depth, max_depth, idx, total, split):
    """Create text stats as an image."""
    h = 120
    w = 300
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:] = (25, 25, 35)

    valid = depth[depth < max_depth * 0.99]
    n_valid = len(valid)
    n_total = depth.size
    pct_valid = 100 * n_valid / max(n_total, 1)

    lines = [
        f"[{split.upper()}] {idx + 1}/{total}",
        f"Shape: {depth.shape[1]}x{depth.shape[0]}",
        f"Range: [{depth.min():.3f}, {depth.max():.3f}] m",
        f"Mean:  {valid.mean():.3f} m" if n_valid > 0 else "Mean: N/A",
        f"Std:   {valid.std():.3f} m" if n_valid > 0 else "Std:  N/A",
        f"Valid: {pct_valid:.1f}% ({n_valid}/{n_total})",
    ]

    for i, text in enumerate(lines):
        color = (200, 200, 220) if i > 0 else (100, 200, 255)
        cv2.putText(canvas, text, (8, 16 + i * 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return canvas


def main():
    args = parse_args()

    # Load metadata
    meta_path = os.path.join(args.data_dir, "metadata.npy")
    if os.path.exists(meta_path):
        meta = np.load(meta_path, allow_pickle=True).item()
        max_depth = args.max_depth or meta.get("max_depth", 5.0)
        print(f"Loaded metadata: {meta}")
    else:
        max_depth = args.max_depth or 5.0
        print(f"No metadata found, using max_depth={max_depth}")

    colormap_cv = COLORMAPS.get(args.colormap, cv2.COLORMAP_VIRIDIS)

    # Load file lists for both splits
    splits = {
        "train": load_file_list(args.data_dir, "train"),
        "test":  load_file_list(args.data_dir, "test"),
    }

    current_split = args.split
    files = splits[current_split]

    if not files:
        print(f"No .npy files found in {args.data_dir}/{current_split}/")
        return

    print(f"Train: {len(splits['train'])} images  |  Test: {len(splits['test'])} images")
    print(f"Viewing: {current_split} ({len(files)} images)")
    print(f"Max depth: {max_depth}m  |  Colormap: {args.colormap}  |  Scale: {args.scale}x\n")
    print("Controls: ←→ navigate  SPACE auto-play  T switch split  "
          "H histogram  S save  R random  ESC quit\n")

    idx = 0
    auto_play = False
    delay = 200  # ms
    show_hist = args.show_histogram

    WIN = "Depth Dataset Viewer"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    while True:
        # Load image
        depth = np.load(files[idx]).astype(np.float32)
        h, w = depth.shape

        # Colorized depth
        bgr = depth_to_bgr(depth, max_depth, colormap_cv)
        bgr_up = cv2.resize(bgr, (w * args.scale, h * args.scale),
                            interpolation=cv2.INTER_NEAREST)
        bgr_up = cv2.flip(bgr_up, 0)  # flip for display (Warp v=0 is bottom)

        # Stats overlay
        stats_img = make_stats_overlay(depth, max_depth, idx, len(files), current_split)
        stats_h, stats_w = stats_img.shape[:2]

        # Resize stats to match display height if needed
        disp_h, disp_w = bgr_up.shape[:2]

        # Build display
        panels = [bgr_up]

        # Side panel: stats + optional histogram
        side_parts = [stats_img]
        if show_hist:
            hist_img = make_histogram(depth, max_depth, hist_w=stats_w)
            side_parts.append(hist_img)

        side_panel = np.vstack(side_parts)
        # Pad side panel height to match depth image
        if side_panel.shape[0] < disp_h:
            pad = np.zeros((disp_h - side_panel.shape[0], side_panel.shape[1], 3), dtype=np.uint8)
            pad[:] = (25, 25, 35)
            side_panel = np.vstack([side_panel, pad])
        elif side_panel.shape[0] > disp_h:
            side_panel = side_panel[:disp_h]

        display = np.hstack([bgr_up, side_panel])

        # Show filename in title
        fname = os.path.basename(files[idx])
        title = f"{WIN}  [{current_split}]  {fname}"
        cv2.setWindowTitle(WIN, title)
        cv2.imshow(WIN, display)

        # Key handling
        wait_ms = delay if auto_play else 0
        key = cv2.waitKey(max(wait_ms, 1)) & 0xFF

        if key == 27 or key == ord('q'):  # ESC / Q
            break

        elif key == ord('d') or key == 83 or key == 3:  # → or D
            idx = (idx + 1) % len(files)

        elif key == ord('a') or key == 81 or key == 2:  # ← or A
            idx = (idx - 1) % len(files)

        elif key == ord(' '):  # SPACE
            auto_play = not auto_play
            print(f"Auto-play: {'ON' if auto_play else 'OFF'}  (delay={delay}ms)")

        elif key == ord('+') or key == ord('='):
            delay = max(10, delay - 50)
            print(f"Delay: {delay}ms")

        elif key == ord('-'):
            delay = min(2000, delay + 50)
            print(f"Delay: {delay}ms")

        elif key == ord('t'):  # Switch split
            current_split = "test" if current_split == "train" else "train"
            files = splits[current_split]
            idx = min(idx, len(files) - 1)
            print(f"Switched to: {current_split} ({len(files)} images)")

        elif key == ord('h'):  # Toggle histogram
            show_hist = not show_hist

        elif key == ord('s'):  # Save
            save_path = f"depth_screenshot_{current_split}_{idx:06d}.png"
            cv2.imwrite(save_path, display)
            print(f"Saved: {save_path}")

        elif key == ord('r'):  # Random
            idx = np.random.randint(0, len(files))

        elif key == ord('g'):  # Go to index
            print("Enter index: ", end="", flush=True)
            try:
                val = int(input())
                idx = max(0, min(val, len(files) - 1))
            except ValueError:
                pass

        elif auto_play:
            idx = (idx + 1) % len(files)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()