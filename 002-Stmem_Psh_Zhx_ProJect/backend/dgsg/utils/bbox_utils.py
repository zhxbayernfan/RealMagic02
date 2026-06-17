import numpy as np
import open3d as o3d

def safe_get_oriented_bbox(pcd: o3d.geometry.PointCloud, min_points=4, eps_jitter=1e-6):
    """
    Robust oriented bbox getter:
    - Remove NaN/Inf points
    - Deduplicate
    - If #points < min_points or rank < 3: fallback to axis-aligned bbox or small box
    - If points are coplanar (rank==2) or collinear (rank==1) we use AABB as fallback.
    """
    # Get numpy points
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        return None

    # Remove NaN/Inf rows
    mask = np.isfinite(pts).all(axis=1)
    pts = pts[mask]
    if pts.shape[0] == 0:
        return None

    # Deduplicate (optional)
    pts_unique = np.unique(pts.round(decimals=6), axis=0)  # round to collapse near duplicates
    if pts_unique.shape[0] < min_points:
        # If too few, try to add tiny jitter to make Qhull happy or fallback
        if pts_unique.shape[0] >= 1:
            # fallback to AABB
            aabb = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(pts_unique))
            # Optionally convert AABB to an oriented box with same center and small extents
            return aabb.get_oriented_bounding_box()
        return None

    # Check rank (dimensionality) using SVD
    centered = pts_unique - pts_unique.mean(axis=0, keepdims=True)
    u, s, vh = np.linalg.svd(centered, full_matrices=False)
    rank = np.sum(s > 1e-6)
    if rank < 3:
        # Points are coplanar or collinear, fallback to AABB
        aabb = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(pts_unique))
        return aabb.get_oriented_bounding_box()

    # Normal path: try OBB but catch exceptions
    try:
        pcd_tmp = o3d.geometry.PointCloud()
        pcd_tmp.points = o3d.utility.Vector3dVector(pts_unique)
        obb = pcd_tmp.get_oriented_bounding_box()
        return obb
    except Exception as e:
        # As last resort, fallback to AABB
        aabb = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(pts_unique))
        return aabb.get_oriented_bounding_box()