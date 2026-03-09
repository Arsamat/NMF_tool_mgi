"""
Helper functions for DEG plots (volcano, MA, label placement).
"""
import numpy as np

# Default plot dimensions used for label placement (Plotly volcano height=500; width ~container)
PLOT_WIDTH_PX = 700
PLOT_HEIGHT_PX = 500
PLOT_MARGIN = 80


def place_labels_no_overlap(points_x, points_y, labels, x_min, x_max, y_min, y_max):
    """
    Compute (ax, ay) in pixels for each label so that label boxes do not overlap.
    points_x, points_y: arrays of data coords. labels: list of str (label lengths used for box width).
    x_min, x_max, y_min, y_max: axis data range for mapping to pixels.
    Returns list of (ax, ay) in pixel offset for each point.
    """
    pw = PLOT_WIDTH_PX - 2 * PLOT_MARGIN
    ph = PLOT_HEIGHT_PX - 2 * PLOT_MARGIN
    if x_max <= x_min:
        x_max = x_min + 1
    if y_max <= y_min:
        y_max = y_min + 1
    x_range = x_max - x_min
    y_range = y_max - y_min

    def data_to_px(x, y):
        x_px = PLOT_MARGIN + (x - x_min) / x_range * pw
        y_px = PLOT_MARGIN + ph - (y - y_min) / y_range * ph  # data y max -> top
        return x_px, y_px

    # Label box size and padding: extra padding to keep arrows/labels clear of each other
    char_w = 8
    label_h = 18
    pad = 14  # larger padding to reduce arrow-vs-label overlap
    gap = 8   # minimum gap between boxes (inflation for overlap check)
    n = len(points_x)
    placed = []  # list of (l, t, r, b) in pixel coords
    anchor_px = []  # (x_px, y_px) of each point for arrow-clearance

    radii = [50, 65, 82, 100, 120, 145, 170]
    angles_deg = [0, 45, 90, 135, 180, 225, 270, 315]

    result = []
    for i in range(n):
        x_px, y_px = data_to_px(points_x[i], points_y[i])
        w = max(48, min(220, len(str(labels[i])) * char_w)) + 2 * pad
        h = label_h + 2 * pad
        ax, ay = None, None
        for r in radii:
            for a in angles_deg:
                theta = np.radians(a)
                ax_c = int(r * np.cos(theta))
                ay_c = int(-r * np.sin(theta))
                box_l = x_px + ax_c
                box_t = y_px + ay_c
                box_r = box_l + w
                box_b = box_t + h
                # Inflate boxes by gap for overlap check (require more separation)
                box_l_gap = box_l - gap
                box_t_gap = box_t - gap
                box_r_gap = box_r + gap
                box_b_gap = box_b + gap
                overlap = False
                for j, (ql, qt, qr, qb) in enumerate(placed):
                    # Require gap between both boxes (inflate placed box too)
                    if not (box_r_gap < ql - gap or qr + gap < box_l_gap or box_b_gap < qt - gap or qb + gap < box_t_gap):
                        overlap = True
                        break
                    # Avoid label covering another point's anchor (arrow start)
                    if j < len(anchor_px):
                        ax_p, ay_p = anchor_px[j]
                        if box_l <= ax_p <= box_r and box_t <= ay_p <= box_b:
                            overlap = True
                            break
                if not overlap:
                    ax, ay = ax_c, ay_c
                    break
            if ax is not None:
                break
        if ax is None:
            ax, ay = 50, -24
        result.append((ax, ay))
        box_l = x_px + ax
        box_t = y_px + ay
        placed.append((box_l, box_t, box_l + w, box_t + h))
        anchor_px.append((x_px, y_px))

    return result
