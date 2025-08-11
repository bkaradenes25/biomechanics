""""
Baseball Pitching Biomechanics Analysis

Extracts:
1. Release Point (X, Y, Z) from .c3d motion capture files
2. Elbow and shoulder joint angles
3. Plots for spatial release point distribution and angle relationships

Data Source: Wasserberger KW, Brady AC, Besky DM, Jones BR, Boddy KJ. The OpenBiomechanics Project: The open source initiative for anonymized, elite-level athletic motion capture data. (2022).
Author: Brendan Karadenes
"""

import numpy as np
import ezc3d
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def detect_throwing_hand(c3d):
    """
    Determine throwing hand by comparing peak finger marker velocities.
    Returns 'RFIN' or 'LFIN'
    """

    marker_labels = c3d['parameters']['POINT']['LABELS']['value']
    points = c3d['data']['points']

    def get_velocity(marker):
        if marker not in marker_labels:
            return 0
        idx = marker_labels.index(marker)
        coords = points[:3, idx, :]
        velocities = np.linalg.norm(np.diff(coords, axis=1), axis = 0)
        return np.max(velocities)

    right_peak = get_velocity("RFIN")
    left_peak = get_velocity("LFIN")

    return "RFIN" if right_peak > left_peak else "LFIN"

def extract_release_point_auto(c3d_path):
    """
    Return peak velocity magnitude for a given marker name.
    """
    c3d = ezc3d.c3d(c3d_path)
    marker_name = detect_throwing_hand(c3d)

    points = c3d['data']['points']
    marker_labels = c3d['parameters']['POINT']['LABELS']['value']
    marker_idx = marker_labels.index(marker_name)

    marker_data = points[:3, marker_idx, :]
    velocities = np.linalg.norm(np.diff(marker_data, axis=1), axis = 0)
    release_frame = np.argmax(velocities)
    release_coords = marker_data[:, release_frame]

    return {
        'file': c3d_path,
        'marker': marker_name,
        'frame': int(release_frame),
        'release_x': release_coords[0],
        'release_y': release_coords[1],
        'release_z': release_coords[2],
    }

def extract_elbow_and_shoulder_angles(file_path):
    """
    Extract the path and coordinates of maximum throwing hand speed.
    """
    c = ezc3d.c3d(file_path)
    markers = c['data']['points'][:3]
    marker_labels = c['parameters']['POINT']['LABELS']['value']

    pitcher_id = Path(file_path).parts[-2]
    prefix = "R" if "RFIN" in marker_labels else "L"

    SHO = markers[:, marker_labels.index(prefix + "SHO"), :]
    ELB = markers[:, marker_labels.index(prefix + "ELB"), :]
    WRA = markers[:, marker_labels.index(prefix + "WRA"), :]
    CLAV_marker = "CLAV" if "CLAV" in marker_labels else "STRN"
    CLAV = markers[:, marker_labels.index(CLAV_marker), :]

    angles = []
    for frame in range(SHO.shape[1]):
        # Elbow angle (between SHO and ELB, and ELB and WRA
        vec1 = SHO[:, frame] - ELB[:, frame]
        vec2 = WRA[:, frame] - ELB[:, frame]
        elbow_angle = np.rad2deg(np.arccos(np.clip(np.dot(vec1, vec2)/ (np.linalg.norm(vec1) * np.linalg.norm(vec2)), -1.0, 1.0)))

        # Shoulder Angle (between CLAV and SHO, and SHO and ELB
        vec3 = CLAV[:, frame] - SHO[:, frame]
        vec4 = SHO[:, frame] - ELB[:, frame]
        shoulder_angle = np.rad2deg(np.arccos(np.clip(np.dot(vec3, vec4) / (np.linalg.norm(vec3) * np.linalg.norm(vec4)), -1.0, 1.0)))

        angles.append({
            "file": file_path,
            "pitcher_id": pitcher_id,
            "frame": frame,
            "elbow_angle": elbow_angle,
            "shoulder_angle": shoulder_angle,
        })

    return angles

# .c3d file directory
base_dir = "openbiomechanics/baseball_pitching/data/c3d"
# Collect all .c3d file paths recursively
c3d_files = list(Path(base_dir).rglob("*.c3d"))
# Store results here
release_data = []
all_angles = []
# Loop through files and extract release points
for file in c3d_files:
    try:
        result = extract_release_point_auto(str(file))

        # parse pitcher ID from path
        result["pitcher_id"] = Path(file).parts[-2]

        release_data.append(result)

        angles = extract_elbow_and_shoulder_angles(str(file))
        all_angles.extend(angles)

    except Exception as e:
        print(f"Error processing {file}: {e}")
# Convert to pandas dataframe
df = pd.DataFrame(release_data)
angle_df = pd.DataFrame(all_angles)

merged_df = df.merge(angle_df, on = ["file", "pitcher_id", "frame"], how = "left")

merged_df.to_csv("release_points_with_elbow_and_shoulder.csv", index = False)

# Side view (z vs. y)
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='release_y', y='release_z', hue='pitcher_id', palette='tab10')
plt.title("Release Point - Side View (Z vs Y)")
plt.xlabel("Release Y (toward/away from catcher)")
plt.ylabel("Release Z (height)")
plt.legend(bbox_to_anchor=(1.05, 1), loc = 'upper left')
plt.tight_layout()
plt.show()

# Top view (x vs. y)
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='release_y', y='release_x', hue='pitcher_id', palette='tab10')
plt.title("Release Point - Side View (X vs Y)")
plt.xlabel("Release Y (toward/away from catcher)")
plt.ylabel("Release X (left/right)")
plt.legend(bbox_to_anchor=(1.05, 1), loc = 'upper left')
plt.tight_layout()
plt.show()

fig = plt.figure(figsize = (10, 7))
ax = fig.add_subplot(111, projection = '3d')
for pid, group in df.groupby("pitcher_id"):
    ax.scatter(group['release_x'], group['release_y'], group['release_z'], label = pid, alpha = 0.7)
ax.set_title("3D Release Point Cloud")
ax.set_xlabel("X (left/right)")
ax.set_ylabel("Y (toward/away)")
ax.set_zlabel("Z (height)")
ax.legend(bbox_to_anchor=(1.1, 1))
plt.tight_layout()
plt.show()

sns.scatterplot(data = merged_df, x = 'shoulder_angle', y = 'release_z', hue = 'pitcher_id', palette = 'tab10')
plt.title("Shoulder Angle vs Release Height (Z)")
plt.xlabel("Shoulder Angle (degrees)")
plt.ylabel("Release Height (Z)")
plt.tight_layout()
plt.show()
