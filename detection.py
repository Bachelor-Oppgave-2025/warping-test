"""Table and ball detection utilities."""

import cv2
import numpy as np

TABLE_HSV_LOWER = np.array([35, 40, 40])
TABLE_HSV_UPPER = np.array([85, 255, 255])

ALLOWED_BALL_LABELS = {
    "cue",
    "yellow",
    "green",
    "brown",
    "blue",
    "pink",
    "black",
    "red",
}
NONGREEN_BALL_LABELS = ALLOWED_BALL_LABELS - {"green"}


def order_points(pts):
    """Order corner points: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

    return rect


def detect_table_corners(frame):
    """
    Detect table corners using contour detection.
    Works with rotated/skewed tables by finding the quadrilateral that best fits the table.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, TABLE_HSV_LOWER, TABLE_HSV_UPPER)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    table_contour = max(contours, key=cv2.contourArea)

    candidates = []
    minrect = _min_rect_corners(table_contour)

    # Candidate 1: polygon approximation.
    epsilon = 0.02 * cv2.arcLength(table_contour, True)
    approx = cv2.approxPolyDP(table_contour, epsilon, True)
    if len(approx) == 4:
        candidates.append(approx.reshape(4, 2).astype("float32"))

    # Candidate 2: contour-extreme corners (robust fallback).
    extreme = _extreme_corners_from_contour(table_contour)
    if extreme is not None:
        candidates.append(extreme)

    if not candidates and minrect is None:
        return None

    # Long-side recordings are much more stable with min-area-rect corners.
    if minrect is not None and candidates:
        long_side_view = any(
            detect_table_orientation(order_points(cand)) for cand in candidates
        )
        if long_side_view:
            return minrect

    if candidates:
        # For short-side, keep perspective-aware corners from contour geometry.
        best = max(candidates, key=_quad_area)
        if _quad_area(best) >= 100:
            return order_points(best)

    return minrect


def detect_table_orientation(corners):
    """
    Detect if table is filmed from short side or long side.
    Returns True if from long side (landscape), False if from short side (portrait).
    """
    # Calculate distances between corners
    # corners order: top-left, top-right, bottom-right, bottom-left
    top_edge = np.linalg.norm(corners[1] - corners[0])  # top side length
    left_edge = np.linalg.norm(corners[3] - corners[0])  # left side length

    # If top edge is longer than left edge, it's landscape (long side view)
    return top_edge > left_edge


_BLOB_DETECTOR = None


def setup_blob_detector():
    """Setup SimpleBlobDetector with parameters optimized for ball detection."""
    params = cv2.SimpleBlobDetector_Params()

    # Filter by area - lenient range
    params.filterByArea = True
    params.minArea = 8
    params.maxArea = 80000

    # Filter by circularity
    params.filterByCircularity = True
    params.minCircularity = 0.05

    # Filter by color (white blobs on mask)
    params.filterByColor = True
    params.blobColor = 255

    # Thresholding for binary/near-binary masks
    params.minThreshold = 0
    params.maxThreshold = 255
    params.thresholdStep = 5

    # Filter by convexity
    params.filterByConvexity = True
    params.minConvexity = 0.2

    # Filter by inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.05

    # Minimum distance between blobs
    params.minDistBetweenBlobs = 2

    return cv2.SimpleBlobDetector_create(params)


def _get_blob_detector():
    return setup_blob_detector()


def detect_nongreen_balls(frame):
    """Detect nongreen balls using foreground masking and circle candidates."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Table mask (green)
    table_mask = cv2.inRange(hsv, TABLE_HSV_LOWER, TABLE_HSV_UPPER)

    # Invert to get non-table objects (balls, shadows, etc.)
    mask = cv2.bitwise_not(table_mask)

    # Apply a light close to connect ball blobs without erasing small balls
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Detect blobs
    detector = _get_blob_detector()
    keypoints = detector.detect(mask)

    detections = []
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        r = max(2, int(kp.size / 2))
        label = classify_ball_color(hsv, x, y, r)
        if label in NONGREEN_BALL_LABELS:
            detections.append({"x": x, "y": y, "r": r, "label": label})

    # Add contour-based circle candidates to catch balls missed by the blob detector.
    min_r, max_r = _estimate_radius_bounds(frame.shape[:2])
    contour_circles = _detect_circles_contour(gray, min_r, max_r)
    detections = _merge_circle_detections(
        detections, contour_circles, hsv, allowed_labels=NONGREEN_BALL_LABELS
    )

    return detections


def detect_green_balls_reflection(frame):
    """Detect green balls from specular highlights and local circular support."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    min_r, max_r = _estimate_radius_bounds(frame.shape[:2])

    highlight_mask = _green_highlight_mask(hsv)
    contours, _ = cv2.findContours(
        highlight_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    best_detection = None
    best_score = 0.0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 1 or area > 24:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] <= 0:
            continue
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        candidate = _fit_green_ball_candidate(gray, hsv, (cx, cy), min_r, max_r)
        if candidate is None:
            continue
        cand_x, cand_y, cand_r = candidate
        if not _inside_playfield_margin(frame.shape[:2], cand_x, cand_y, cand_r):
            continue
        green_score = _green_ring_score(hsv, cand_x, cand_y, cand_r)
        highlight_ratio = _highlight_ratio(hsv, cand_x, cand_y, cand_r)
        if green_score < 0.42:
            continue
        if highlight_ratio < 0.01 or highlight_ratio > 0.18:
            continue

        score = (green_score * 1.5) - abs(highlight_ratio - 0.06)
        if score > best_score:
            best_score = score
            best_detection = {
                "x": int(cand_x),
                "y": int(cand_y),
                "r": int(cand_r),
                "label": "green",
            }

    if best_detection is None:
        return []
    return [best_detection]


def detect_balls(frame):
    """Detect balls from an inner-table ROI, Sobel circles, and HSV voting."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    min_r, max_r = _estimate_radius_bounds(frame.shape[:2])

    roi_mask = _playfield_roi_mask(frame.shape[:2])
    circle_candidates, edge_image = _detect_circles_sobel(gray, roi_mask, min_r, max_r)

    detections = []
    for x, y, r in circle_candidates:
        if not _inside_playfield_margin(frame.shape[:2], x, y, r):
            continue
        if _edge_ring_score(edge_image, x, y, r) < 0.12:
            continue

        label = classify_ball_color(hsv, x, y, r)
        support = _label_support_ratio(hsv, x, y, r, label)
        if not _accept_label_candidate(label, support, hsv, (x, y, r)):
            continue

        detections.append({"x": int(x), "y": int(y), "r": int(r), "label": label})

    # Keep the older nongreen detector as a fallback for weak-circle cases.
    fallback = detect_nongreen_balls(frame)
    return _merge_ball_detections(detections, fallback)


def detect_balls_blob(frame):
    """Backward-compatible wrapper for the current ball detector."""
    return detect_balls(frame)


def detect_balls_hough(_frame):
    """Deprecated: kept for reference, prefer detect_balls_blob."""
    return []


def detect_pockets_warp(frame):
    """Return fixed pocket boxes from warp geometry (4 corners + 2 middle long-side)."""
    h, w = frame.shape[:2]
    max_size = max(8, min(h, w) - 2)
    size = min(max(12, int(min(h, w) * 0.07)), max_size)
    half = size // 2
    x_min = half
    y_min = half
    x_max = max(x_min, w - 1 - half)
    y_max = max(y_min, h - 1 - half)
    x_mid = int(np.clip(w // 2, x_min, x_max))
    y_mid = int(np.clip(h // 2, y_min, y_max))

    # Four corners of the warped frame.
    centers = [
        (x_min, y_min),
        (x_max, y_min),
        (x_max, y_max),
        (x_min, y_max),
    ]

    # Two middle pockets on the long sides.
    if w >= h:
        centers.extend([(x_mid, y_min), (x_mid, y_max)])
    else:
        centers.extend([(x_min, y_mid), (x_max, y_mid)])

    pockets = []
    for cx, cy in centers:
        x = int(np.clip(cx - half, 0, max(0, w - size)))
        y = int(np.clip(cy - half, 0, max(0, h - size)))
        pockets.append({"x": x, "y": y, "size": int(size)})

    return pockets


def classify_ball_color(hsv_frame, x, y, r):
    """Classify a ball color based on mean HSV in a circular mask."""
    h, s, v = _median_hsv_in_circle(hsv_frame, x, y, r)
    label = "unknown"

    if v < 50:
        label = "black"
    elif s < 40 and v > 150:
        label = "cue"
    elif 130 <= h <= 179 and 190 <= v <= 255:
        label = "pink"
    elif 0 <= h <= 50 and 0 <= s <= 150 and 0 <= v <= 180:
        label = "brown"
    elif 90 <= h <= 140 and 150 <= s <= 255 and 150 <= v <= 200:
        label = "blue"
    elif 80 <= h <= 100 and 150 <= s <= 255:
        label = "green"
    elif 15 <= h <= 40:
        label = "yellow"
    elif h <= 15 or h >= 165:
        label = "red"

    return label


def _green_highlight_mask(hsv):
    """Return candidate highlight blobs likely caused by green-ball reflections."""
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    val_blur = cv2.GaussianBlur(val, (0, 0), 5)
    contrast = cv2.subtract(val, val_blur)

    bright_mask = cv2.inRange(val, 205, 255)
    low_sat_mask = cv2.inRange(sat, 0, 80)
    contrast_mask = cv2.inRange(contrast, 18, 255)
    mask = cv2.bitwise_and(bright_mask, low_sat_mask)
    mask = cv2.bitwise_and(mask, contrast_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


def _playfield_roi_mask(shape):
    """Return an inner rectangle mask to ignore rails, pockets, and frame edges."""
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    margin_x = max(14, int(w * 0.09))
    margin_y = max(14, int(h * 0.09))
    x1 = margin_x
    y1 = margin_y
    x2 = max(x1 + 1, w - margin_x)
    y2 = max(y1 + 1, h - margin_y)
    mask[y1:y2, x1:x2] = 255
    return mask


def _median_hsv_in_circle(hsv_frame, x, y, r):
    h, w = hsv_frame.shape[:2]
    x = int(np.clip(x, 0, w - 1))
    y = int(np.clip(y, 0, h - 1))
    r = max(2, int(r * 0.6))

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (x, y), r, 255, -1)

    pixels = hsv_frame[mask == 255]
    if pixels.size == 0:
        return 0.0, 0.0, 0.0

    # Filter out very dark pixels to avoid shadow bias.
    val = pixels[:, 2]
    keep = val > 30
    if keep.any():
        pixels = pixels[keep]

    h_med = float(np.median(pixels[:, 0]))
    s_med = float(np.median(pixels[:, 1]))
    v_med = float(np.median(pixels[:, 2]))
    return h_med, s_med, v_med


def _detect_circles_contour(gray, min_radius, max_radius):
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circles = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area <= 0:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        if circularity < 0.5:
            continue

        (x, y), radius = cv2.minEnclosingCircle(contour)
        radius = int(radius)
        if min_radius <= radius <= max_radius:
            circles.append((int(x), int(y), radius))

    return circles


def _detect_circles_sobel(gray, roi_mask, min_radius, max_radius):
    """Detect circular candidates on a Sobel edge image inside the playfield ROI."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    grad_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)
    grad_mag = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX)
    edge_image = grad_mag.astype(np.uint8)
    edge_image = cv2.bitwise_and(edge_image, edge_image, mask=roi_mask)

    circles = cv2.HoughCircles(
        edge_image,
        cv2.HOUGH_GRADIENT,
        dp=1.15,
        minDist=max(6, int(min_radius * 1.8)),
        param1=120,
        param2=10,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    candidates = []
    if circles is not None:
        for x, y, r in np.round(circles[0, :]).astype(int):
            if (
                roi_mask[
                    int(np.clip(y, 0, roi_mask.shape[0] - 1)),
                    int(np.clip(x, 0, roi_mask.shape[1] - 1)),
                ]
                == 0
            ):
                continue
            if _is_near_existing(
                [{"x": cx, "y": cy, "r": cr} for cx, cy, cr in candidates], x, y, r
            ):
                continue
            candidates.append((int(x), int(y), int(r)))

    return candidates, edge_image


def _fit_green_ball_candidate(gray, hsv, center, min_radius, max_radius):
    """Fit a local circle around a highlight blob and validate its green surround."""
    cx, cy = center
    h, w = gray.shape[:2]
    pad = max_radius * 3
    x1 = max(0, cx - pad)
    y1 = max(0, cy - pad)
    x2 = min(w, cx + pad + 1)
    y2 = min(h, cy + pad + 1)
    roi = gray[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    roi = cv2.GaussianBlur(roi, (5, 5), 0)
    circles = cv2.HoughCircles(
        roi,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(4, min_radius),
        param1=80,
        param2=10,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is None:
        return None

    best = None
    best_score = -1.0
    for circle in np.round(circles[0, :]).astype(int):
        x, y, r = circle
        full_x = x + x1
        full_y = y + y1
        center_dist_sq = float((full_x - cx) ** 2 + (full_y - cy) ** 2)
        if center_dist_sq > max(4.0, float(r * r)):
            continue
        if not _inside_playfield_margin(gray.shape[:2], full_x, full_y, r):
            continue

        green_score = _green_ring_score(hsv, full_x, full_y, r)
        score = green_score - (center_dist_sq / max(1.0, float(r * r)))
        if score > best_score:
            best_score = score
            best = (full_x, full_y, r)

    return best


def _estimate_radius_bounds(shape):
    h, w = shape[:2]
    min_dim = min(h, w)
    min_radius = max(2, int(min_dim * 0.01))
    max_radius = max(min_radius + 2, int(min_dim * 0.045))
    return min_radius, max_radius


def _merge_circle_detections(detections, circles, hsv, allowed_labels=None):
    merged = list(detections)
    if allowed_labels is None:
        allowed_labels = ALLOWED_BALL_LABELS
    for x, y, r in circles:
        if _is_near_existing(merged, x, y, r):
            continue
        label = classify_ball_color(hsv, x, y, r)
        if label in allowed_labels:
            merged.append({"x": int(x), "y": int(y), "r": int(r), "label": label})
    return merged


def _merge_ball_detections(primary, secondary):
    merged = list(primary)
    for det in secondary:
        if not _is_near_existing(merged, det["x"], det["y"], det["r"]):
            merged.append(det)
    return merged


def _is_near_existing(detections, x, y, r):
    for det in detections:
        dx = det["x"] - x
        dy = det["y"] - y
        if dx * dx + dy * dy <= (r * r):
            return True
    return False


def _green_ring_score(hsv_frame, x, y, r):
    """Measure how much of a ring around the candidate looks like green ball surface."""
    h, w = hsv_frame.shape[:2]
    mask_outer = np.zeros((h, w), dtype=np.uint8)
    mask_inner = np.zeros((h, w), dtype=np.uint8)

    outer_r = max(3, int(r))
    inner_r = max(1, int(r * 0.45))
    cv2.circle(mask_outer, (int(x), int(y)), outer_r, 255, -1)
    cv2.circle(mask_inner, (int(x), int(y)), inner_r, 255, -1)
    ring_mask = cv2.subtract(mask_outer, mask_inner)

    pixels = hsv_frame[ring_mask == 255]
    if pixels.size == 0:
        return 0.0

    hue = pixels[:, 0]
    sat = pixels[:, 1]
    val = pixels[:, 2]
    green = (hue >= 55) & (hue <= 95) & (sat >= 35) & (val >= 35)
    return float(np.count_nonzero(green)) / float(len(pixels))


def _edge_ring_score(edge_image, x, y, r):
    """Return the average normalized edge strength on a thin ring around a candidate."""
    h, w = edge_image.shape[:2]
    mask_outer = np.zeros((h, w), dtype=np.uint8)
    mask_inner = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask_outer, (int(x), int(y)), max(3, int(r * 1.05)), 255, -1)
    cv2.circle(mask_inner, (int(x), int(y)), max(1, int(r * 0.75)), 255, -1)
    ring = cv2.subtract(mask_outer, mask_inner)
    pixels = edge_image[ring == 255]
    if pixels.size == 0:
        return 0.0
    return float(np.mean(pixels)) / 255.0


def _ball_pixels(hsv_frame, x, y, r, scale=0.65):
    h, w = hsv_frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (int(x), int(y)), max(2, int(r * scale)), 255, -1)
    pixels = hsv_frame[mask == 255]
    if pixels.size == 0:
        return np.empty((0, 3), dtype=hsv_frame.dtype)
    return pixels


def _label_support_ratio(hsv_frame, x, y, r, label):
    """Return how much of the candidate interior supports the assigned color label."""
    pixels = _ball_pixels(hsv_frame, x, y, r)
    if pixels.size == 0:
        return 0.0

    hue = pixels[:, 0]
    sat = pixels[:, 1]
    val = pixels[:, 2]

    if label == "black":
        keep = val < 70
    elif label == "cue":
        keep = (sat < 55) & (val > 150)
    elif label == "brown":
        keep = (
            (hue >= 0)
            & (hue <= 50)
            & (sat >= 0)
            & (sat <= 150)
            & (val >= 0)
            & (val <= 180)
        )
    elif label == "pink":
        keep = (hue >= 130) & (hue <= 179) & (val >= 190) & (val <= 255)
    elif label == "yellow":
        keep = (hue >= 15) & (hue <= 40)
    elif label == "green":
        keep = (hue >= 80) & (hue <= 100) & (sat >= 150) & (sat <= 255)
    elif label == "blue":
        keep = (
            (hue >= 90)
            & (hue <= 140)
            & (sat >= 150)
            & (sat <= 255)
            & (val >= 150)
            & (val <= 200)
        )
    elif label == "red":
        keep = (hue <= 15) | (hue >= 165)
    else:
        return 0.0

    return float(np.count_nonzero(keep)) / float(len(pixels))


def _accept_label_candidate(label, support, hsv_frame, ball):
    """Apply simple voting thresholds to reject false positives after classification."""
    x, y, r = ball
    if label not in ALLOWED_BALL_LABELS:
        return False

    accepted = False
    if label == "green":
        accepted = support >= 0.16 and _green_ring_score(hsv_frame, x, y, r) >= 0.20
    elif label == "cue":
        accepted = support >= 0.30
    elif label == "black":
        accepted = support >= 0.40
    elif label in {"yellow", "blue", "pink", "brown"}:
        accepted = support >= 0.14
    elif label == "red":
        accepted = support >= 0.16
    return accepted


def _highlight_ratio(hsv_frame, x, y, r):
    """Return the fraction of bright low-saturation pixels inside the candidate ball."""
    h, w = hsv_frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (int(x), int(y)), max(2, int(r * 0.75)), 255, -1)
    pixels = hsv_frame[mask == 255]
    if pixels.size == 0:
        return 0.0

    sat = pixels[:, 1]
    val = pixels[:, 2]
    highlight = (val >= 205) & (sat <= 80)
    return float(np.count_nonzero(highlight)) / float(len(pixels))


def _inside_playfield_margin(shape, x, y, r):
    """Reject edge and pocket-adjacent detections where reflections are unstable."""
    h, w = shape[:2]
    margin = max(12, int(r * 2.2))
    return margin <= x < (w - margin) and margin <= y < (h - margin)


def _min_rect_corners(contour):
    if contour is None or len(contour) < 4:
        return None
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect).astype(np.float32)
    if _quad_area(box) < 100:
        return None
    return order_points(box)


def _extreme_corners_from_contour(contour):
    pts = contour.reshape(-1, 2)
    if pts.shape[0] < 4:
        return None

    sums = pts[:, 0] + pts[:, 1]
    diffs = pts[:, 0] - pts[:, 1]

    corners = np.array(
        [
            pts[np.argmin(sums)],  # top-left
            pts[np.argmax(diffs)],  # top-right
            pts[np.argmax(sums)],  # bottom-right
            pts[np.argmin(diffs)],  # bottom-left
        ],
        dtype=np.float32,
    )

    if _quad_area(corners) < 100:
        return None
    return corners


def _quad_area(pts):
    ordered = order_points(np.array(pts, dtype=np.float32))
    x = ordered[:, 0]
    y = ordered[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return float(area)
