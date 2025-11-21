""" Load pointcloud in interactive marking mode
User marks the four corners and presses Q: when done
Repeat this for all squares.
"""
import pandas as pd
import open3d as o3d
import numpy as np
import copy
import os
import re


def pick_points(pcd: o3d.geometry.PointCloud):
    """ Visualize the pointcloud and let the user pick points.
     Returns the list of point indices picked by the user."""
    print("")
    print(
        "1) Please pick at least three correspondences using [shift + left click]"
    )
    print("   Press [shift + right click] to undo point picking")
    print("2) Afther picking points, press q for close the window")
    print("3) Use shift + '+' or '-' to change the marker size")
    # For the other controls see:
    # https://github.com/isl-org/Open3D/blob/d7341c4373e50054d9dbe28ed84c09bb153de2f8/src/Visualization/Visualizer/VisualizerWithEditing.cpp#L124

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    return vis.get_picked_points()


def extract_colors(pcd: o3d.geometry.PointCloud, marked_points: list, z_padding: float = 0.01):
    """ Extract RGB values of all points inside the 3D box defined by the
        square (XY from the four selected points) extruded in Z by
        z_padding meters on both sides (default 0.01 = 1 cm).

        Args:
            pcd: Open3D point cloud
            marked_points: list of four 3D corner coordinates (from picked points)
            z_padding: padding in meters to extrude above/below the square in Z

        Returns:
            inside_points: (N,3) array of point XYZ inside the box
            inside_colors: (N,3) array of RGB colors (float in [0,1])
            true_indexes: array of original point indices selected
    """
    # Convert marked corner coordinates to numpy array
    square_corners = np.array(marked_points)  # shape (4,3)
    square_corners_2d = square_corners[:, :2]  # Only X and Y

    # Get all points from the pointcloud (don't rely on a global variable)
    points = np.asarray(pcd.points)
    points_2d = points[:, :2]

    # Axis-aligned XY bounds from the four corners
    min_xy = np.min(square_corners_2d, axis=0)
    max_xy = np.max(square_corners_2d, axis=0)

    # Z bounds: use min/max Z from the picked corners and expand by padding
    min_z = np.min(square_corners[:, 2])
    max_z = np.max(square_corners[:, 2])
    min_z_ext = min_z - z_padding
    max_z_ext = max_z + z_padding

    # Mask points that are inside the XY box AND inside the Z range
    inside_xy = np.all((points_2d >= min_xy) & (points_2d <= max_xy), axis=1)
    inside_z = (points[:, 2] >= min_z_ext) & (points[:, 2] <= max_z_ext)
    inside_mask = inside_xy & inside_z

    colors = np.asarray(pcd.colors)
    inside_points = points[inside_mask]
    inside_colors = colors[inside_mask]
    true_indexes = np.where(inside_mask)[0]
    return inside_points, inside_colors, true_indexes


def generate_topview_image(pcd: o3d.geometry.PointCloud, selected_indexes: list, filename: str):
    """ Render a topview (XY) of the pointcloud using Open3D.
      marking selected points purple
    """
    # Colors: base pink and highlight purple
    pink = np.array([1.0, 0.0, 1.0])
    purple = np.array([0.5, 0.0, 0.5])

    # Build a full-color numpy array, paint all pink then override selected points with purple
    if len(np.asarray(pcd.points)) == 0:
        # nothing to show
        return

    colors = np.asarray(pcd.colors)
    if colors.size == 0:
        # initialize colors if not present
        colors = np.tile(pink, (len(np.asarray(pcd.points)), 1))
    else:
        colors = colors.copy()
        colors[:] = pink

    if len(selected_indexes) > 0:
        colors[selected_indexes] = purple

    pcd.colors = o3d.utility.Vector3dVector(colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Topview (XY) with Selected Points',
                      width=800,
                      height=800, visible=True)
    vis.add_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(f'{filename}.png')
    vis.run()
    print(f'Saved Open3D topview image as {filename}.png')
    vis.destroy_window()


if __name__ == '__main__':
    file = "Raw_data/f00127_20251025T210329_full_sx000_sy002.ply"
    pcd = o3d.io.read_point_cloud(file)
    pcd_copy = copy.deepcopy(pcd)

    def extract_timestamp_from_filename(filepath: str) -> str:
        """Extract T{HHMMSS} from filename. Returns string like 'T010330' or empty string if not found."""
        base = os.path.basename(filepath)
        m = re.search(r'T(\d{6})', base)
        if m:
            return 'T' + m.group(1)
        return ''

    while True:
        pcd = copy.deepcopy(pcd_copy)
        points = np.asarray(pcd.points)
        square = pick_points(pcd)

        marked_points = []
        for point_idx in square:
            point = points[point_idx, :]
            marked_points.append(point.tolist())
        print("point_ids:", square)
        print("coordinates: ", marked_points)

        if len(square) != 4:
            print("Warning: not exactly four points selected, \
                  cannot process this as a square!")
            exit(1)

        # Use a 3D box selection: extrude the selected square by +/- 1 cm in Z
        inside_points, inside_colors, selected_indexes = extract_colors(
            pcd, marked_points, z_padding=3
        )

        # Ask user for filename (e.g., A1) and automatically append timestamp
        user_input = input(
            "Enter filename to save RGB values (squares are named from A1-D6): "
        ).strip()

        # Normalize user input (remove extension if provided)
        if user_input:
            base_name = os.path.splitext(user_input)[0]
        else:
            base_name = "output"

        ts = extract_timestamp_from_filename(file)
        if ts:
            out_name = f"{base_name}-{ts}"
        else:
            out_name = base_name

        # Save topview image and CSV using out_name
        generate_topview_image(pcd, selected_indexes, out_name)
        df = pd.DataFrame(inside_colors, columns=["R", "G", "B"])
        df.to_csv(f"{out_name}.csv", index=False)
        print(f"Saved RGB values to {out_name}.csv")
