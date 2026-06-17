"""Motion-based keyframe filter using ORB + optical flow.

Compares each frame against the last saved frame, not the last raw frame.
Two-layer detection:
  Layer 1: ORB feature matching + essential matrix decomposition (precise)
  Layer 2: Lucas-Kanade sparse optical flow (fallback when ORB features insufficient)
"""

import numpy as np
import cv2


class MotionDetector:
    """Two-layer motion detector for keyframe selection."""

    def __init__(self, fx, fy, cx, cy, trans_thresh=10.0, rot_thresh=3.0,
                 flow_fallback_thresh=2.0):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.trans_thresh = trans_thresh
        self.rot_thresh = rot_thresh
        self.flow_fallback_thresh = flow_fallback_thresh

        self.orb = cv2.ORB_create(nfeatures=1000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)

        self.ref_gray = None
        self.ref_kp = None
        self.ref_des = None

        self.layer1_saves = 0
        self.layer1_skips = 0
        self.layer2_saves = 0
        self.layer2_skips = 0

    def _compute_rotation_angle(self, R):
        cos_angle = np.clip((float(np.trace(R)) - 1.0) / 2.0, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))

    def _layer1_orb(self, curr_gray):
        kp2, des2 = self.orb.detectAndCompute(curr_gray, None)
        if des2 is None or len(des2) < 8 or self.ref_des is None:
            return None

        matches = self.bf.knnMatch(self.ref_des, des2, k=2)
        good = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.75 * n.distance:
                    good.append(m)

        if len(good) < 8:
            return None

        pts1 = np.float32([self.ref_kp[m.queryIdx].pt for m in good])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

        E, mask = cv2.findEssentialMat(
            pts1, pts2, focal=self.fx, pp=(self.cx, self.cy),
            method=cv2.RANSAC, prob=0.999, threshold=1.0
        )
        if E is None or mask is None or E.shape != (3, 3):
            return None

        _, R, _, _ = cv2.recoverPose(E, pts1, pts2, focal=self.fx, pp=(self.cx, self.cy))

        inlier_mask = mask.flatten() == 1
        if inlier_mask.sum() < 5:
            return None

        inlier_pts1 = pts1[inlier_mask]
        inlier_pts2 = pts2[inlier_mask]
        displacements = np.linalg.norm(inlier_pts2 - inlier_pts1, axis=1)
        median_disp = np.median(displacements)

        angle_deg = self._compute_rotation_angle(R)
        should_save = (median_disp > self.trans_thresh) or (angle_deg > self.rot_thresh)
        return should_save

    def _layer2_flow(self, curr_gray):
        pts = cv2.goodFeaturesToTrack(
            self.ref_gray, maxCorners=200, qualityLevel=0.01, minDistance=7
        )
        if pts is None or len(pts) < 10:
            return False

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.ref_gray, curr_gray, pts, None
        )
        valid = status.flatten() == 1
        if valid.sum() == 0:
            return False

        flow_mag = np.linalg.norm(next_pts[valid] - pts[valid], axis=1)
        avg_flow = flow_mag.mean()
        return avg_flow > self.flow_fallback_thresh

    def _update_ref(self, gray):
        self.ref_gray = gray.copy()
        self.ref_kp, self.ref_des = self.orb.detectAndCompute(gray, None)

    def should_save(self, curr_gray):
        """Returns (should_save: bool, layer: int). layer: 0=first, 1=ORB, 2=flow."""
        if self.ref_gray is None:
            self._update_ref(curr_gray)
            return True, 0

        result = self._layer1_orb(curr_gray)
        if result is not None:
            if result:
                self.layer1_saves += 1
                self._update_ref(curr_gray)
            else:
                self.layer1_skips += 1
            return result, 1

        should = self._layer2_flow(curr_gray)
        if should:
            self.layer2_saves += 1
            self._update_ref(curr_gray)
        else:
            self.layer2_skips += 1
        return should, 2
