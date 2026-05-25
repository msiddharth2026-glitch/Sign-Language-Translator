"""
Geometric hand feature extractor.
Produces a rotation/scale/translation invariant feature vector from 21 landmarks.
"""
import numpy as np

# MediaPipe landmark indices
WRIST       = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP         = 1,2,3,4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP         = 5,6,7,8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP     = 9,10,11,12
RING_MCP,   RING_PIP,   RING_DIP,   RING_TIP       = 13,14,15,16
PINKY_MCP,  PINKY_PIP,  PINKY_DIP,  PINKY_TIP      = 17,18,19,20

FINGER_TIPS  = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
FINGER_MCPS  = [THUMB_MCP, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]
FINGER_PIPS  = [THUMB_IP,  INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP]


def _pts(lm):
    """Convert landmark list to Nx3 numpy array."""
    return np.array([[l.x, l.y, l.z] for l in lm], dtype=np.float32)


def normalize_landmarks(pts):
    """
    Translate to wrist origin, scale by palm size, rotate to align
    index-MCP → middle-MCP axis with x-axis.
    Returns 63-dim normalised vector.
    """
    pts = pts.copy()
    # 1. Translate: wrist to origin
    pts -= pts[WRIST]
    # 2. Scale: palm diagonal = 1
    palm_size = np.linalg.norm(pts[MIDDLE_MCP] - pts[WRIST]) + 1e-6
    pts /= palm_size
    # 3. Rotate in XY plane: align wrist→middle_MCP to +Y axis
    ref = pts[MIDDLE_MCP, :2]
    angle = np.arctan2(ref[0], ref[1])   # angle to rotate to +Y
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, -s], [s, c]])
    pts[:, :2] = pts[:, :2] @ rot.T
    return pts.flatten()


def finger_angles(pts):
    """
    Compute bend angle at each PIP joint for 4 fingers + thumb IP.
    Returns 5 values in [0,1] (0=straight, 1=fully bent).
    """
    joints = [
        (THUMB_MCP,  THUMB_IP,   THUMB_TIP),
        (INDEX_MCP,  INDEX_PIP,  INDEX_TIP),
        (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP),
        (RING_MCP,   RING_PIP,   RING_TIP),
        (PINKY_MCP,  PINKY_PIP,  PINKY_TIP),
    ]
    angles = []
    for a, b, c in joints:
        v1 = pts[a] - pts[b]
        v2 = pts[c] - pts[b]
        n1 = np.linalg.norm(v1) + 1e-6
        n2 = np.linalg.norm(v2) + 1e-6
        cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
        angle = np.arccos(cos_a) / np.pi   # 0=straight, 1=180°
        angles.append(angle)
    return np.array(angles, dtype=np.float32)


def finger_openness(pts):
    """
    Ratio of tip-to-wrist distance vs MCP-to-wrist distance.
    >1 = extended, <1 = curled.
    """
    vals = []
    for tip, mcp in zip(FINGER_TIPS, FINGER_MCPS):
        d_tip = np.linalg.norm(pts[tip] - pts[WRIST])
        d_mcp = np.linalg.norm(pts[mcp] - pts[WRIST]) + 1e-6
        vals.append(d_tip / d_mcp)
    return np.array(vals, dtype=np.float32)


def fingertip_distances(pts):
    """Pairwise distances between all 5 fingertips (10 values)."""
    tips = pts[FINGER_TIPS]
    dists = []
    for i in range(5):
        for j in range(i+1, 5):
            dists.append(np.linalg.norm(tips[i] - tips[j]))
    return np.array(dists, dtype=np.float32)


def palm_normal(pts):
    """Normal vector of palm plane (3 values)."""
    v1 = pts[INDEX_MCP] - pts[WRIST]
    v2 = pts[PINKY_MCP] - pts[WRIST]
    n  = np.cross(v1, v2)
    norm = np.linalg.norm(n) + 1e-6
    return (n / norm).astype(np.float32)


def thumb_index_angle(pts):
    """Angle between thumb and index finger vectors."""
    v_thumb = pts[THUMB_TIP] - pts[THUMB_MCP]
    v_index = pts[INDEX_TIP] - pts[INDEX_MCP]
    n1 = np.linalg.norm(v_thumb) + 1e-6
    n2 = np.linalg.norm(v_index) + 1e-6
    cos_a = np.clip(np.dot(v_thumb, v_index) / (n1 * n2), -1, 1)
    return np.array([np.arccos(cos_a) / np.pi], dtype=np.float32)


def extract_features(lm):
    """
    Full feature vector from MediaPipe landmark list.
    Returns 63 + 5 + 5 + 10 + 3 + 1 = 87-dim vector.
    """
    pts = _pts(lm)
    norm_lm  = normalize_landmarks(pts)          # 63
    f_angles = finger_angles(pts)                # 5
    f_open   = finger_openness(pts)              # 5
    tip_dist = fingertip_distances(pts)          # 10
    p_normal = palm_normal(pts)                  # 3
    ti_angle = thumb_index_angle(pts)            # 1
    return np.concatenate([norm_lm, f_angles, f_open, tip_dist, p_normal, ti_angle])
