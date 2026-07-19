#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Description: Place the car in the known accelearation map and relocalize it.
"""
from __future__ import annotations

import numpy as np

from fsd_path_planning.relocalization.relocalization_base_class import (
    RelocalizationCallable,
    Relocalizer,
)
from fsd_path_planning.types import FloatArray
from fsd_path_planning.utils.math_utils import rotate


# Gates for a usable row fit: the initial fit locks the run's starting
# direction, and a questionable fit must be rejected (and retried with more
# cones) rather than accepted.
INLIER_THRESHOLD_M = 0.35
MIN_INLIER_CONES = 4
MIN_INLIER_SPAN_M = 10.0
MAX_ANGLE_TO_HEADING_RAD = np.deg2rad(10.0)

# GENTLE-TRIM MODE (current): the initial path is a straight line from the
# staging point along the car's heading (initial steering ~0, the fit does
# NOT set the start vector — an early bad fit steered the car hard once).
# While driving, the path trims toward the measured cone rows with three
# layers of protection so it can only fix slight missteering:
#   1. tiny per-cycle steps (direction and lateral),
#   2. a hard TOTAL cap relative to the initial heading / staging spot,
#   3. fits that disagree wildly with the current path are ignored.
MAX_DIRECTION_STEP_RAD = np.deg2rad(0.15)
MAX_TOTAL_DIRECTION_RAD = np.deg2rad(3.0)
MAX_FIT_DISAGREEMENT_RAD = np.deg2rad(5.0)
MAX_LATERAL_STEP_M = 0.03
MAX_TOTAL_LATERAL_M = 0.5
MIN_ROW_INLIERS_FOR_LATERAL = 3
MAX_SPAN_REGRESSION_M = 5.0
# the lane is 3 m wide; if the two fitted intercepts are not roughly that far
# apart, the fitter has split a single visible row into "two rows" and the
# centerline is garbage — do not use it for lateral trim
MIN_LANE_WIDTH_M = 2.0
MAX_LANE_WIDTH_M = 4.0


def fit_two_parallel_rows(
    points: FloatArray, initial_is_left: np.ndarray
) -> tuple[float, float, float, np.ndarray, np.ndarray] | None:
    """
    Fit two parallel lines (shared slope, one intercept per cone row) through
    the cone rows using least squares. Row membership and outliers are
    re-estimated every iteration, so cones misassigned by the initial guess
    (e.g. because the heading estimate is off) migrate to the correct row.

    Parameters:
    - points: Nx2 array of cone candidates (local frame).
    - initial_is_left: boolean array, initial row assignment guess.

    Returns:
    - (slope, intercept_left, intercept_right, is_left, inlier_mask) where the
      masks cover `points`, or None if the fit is degenerate.
    """
    is_left = initial_is_left.copy()
    inlier = np.ones(len(points), dtype=bool)
    coefficients = np.zeros(3)
    for _ in range(10):
        pts = points[inlier]
        if len(pts) < 2:
            return None
        # columns: slope, left intercept, right intercept
        design = np.column_stack(
            (
                pts[:, 0],
                is_left[inlier].astype(float),
                (~is_left[inlier]).astype(float),
            )
        )
        coefficients, *_ = np.linalg.lstsq(design, pts[:, 1], rcond=None)
        slope = float(coefficients[0])

        residuals_left = np.abs(points[:, 1] - (slope * points[:, 0] + coefficients[1]))
        residuals_right = np.abs(points[:, 1] - (slope * points[:, 0] + coefficients[2]))
        new_is_left = residuals_left <= residuals_right
        new_inlier = np.minimum(residuals_left, residuals_right) < INLIER_THRESHOLD_M
        if new_inlier.sum() < 2:
            return None
        if (new_inlier == inlier).all() and (new_is_left == is_left).all():
            break
        inlier = new_inlier
        is_left = new_is_left

    return (
        float(coefficients[0]),
        float(coefficients[1]),
        float(coefficients[2]),
        is_left,
        inlier,
    )


class AccelerationRelocalizer(Relocalizer):
    # human-readable outcome of the most recent fit attempt, for logging by
    # the pathing node
    last_report: str = "no attempt yet"

    def __init__(self):
        super().__init__()
        # path model in the map frame: the known path (a straight line along
        # +x) rotated by _theta, shifted by _lateral to path-left, anchored at
        # _anchor
        self._theta: float = 0.0
        self._initial_theta: float = 0.0
        self._lateral: float = 0.0
        self._anchor: FloatArray | None = None
        self._tracked_span: float = 0.0
        self._last_report_bucket = None
        self._last_reject: str = ""

    def _build_transforms(
        self,
    ) -> tuple[RelocalizationCallable, RelocalizationCallable]:
        theta = self._theta
        lateral = np.array([0.0, self._lateral])
        anchor = self._anchor.copy()

        def transform_to_known_frame(position_2d, yaw):
            return rotate(position_2d - anchor, -theta) - lateral, yaw - theta

        def transform_to_base_frame(position_2d, yaw):
            return rotate(position_2d + lateral, theta) + anchor, yaw + theta

        return transform_to_known_frame, transform_to_base_frame

    def attempt_relocalization_calculation(
        self,
        cones: list[FloatArray],
        vehicle_position: FloatArray,
        vehicle_direction: FloatArray,
    ) -> None:
        if not self.is_relocalized:
            super().attempt_relocalization_calculation(
                cones, vehicle_position, vehicle_direction
            )
            return

        # already relocalized: gently trim the path toward the measured cone
        # rows (rate-limited and hard-capped, see constants above)
        self._track(cones, vehicle_position, vehicle_direction)

    def _fit_rows(
        self,
        cones: list[FloatArray],
        vehicle_position: FloatArray,
        vehicle_direction: FloatArray,
    ) -> dict | None:
        """Fit the two cone rows in the car's local frame. Returns the fit
        summary, or None (with self._last_reject set) if unusable."""
        all_cones = np.row_stack(cones)

        if len(all_cones) < MIN_INLIER_CONES:
            self._last_reject = (
                f"rejected: {len(all_cones)} cones in map (<{MIN_INLIER_CONES})"
            )
            return None

        vehicle_yaw = np.arctan2(*vehicle_direction[::-1])
        all_cones_local = rotate(all_cones - vehicle_position, -vehicle_yaw)

        # the lane is 3 m wide, so the rows are expected around y = +-1.5 m
        candidates = all_cones_local[np.abs(all_cones_local[:, 1]) < 2.5]
        if len(candidates) < MIN_INLIER_CONES:
            self._last_reject = (
                f"rejected: {len(candidates)}/{len(all_cones)} cones in the "
                f"|y|<2.5 band (<{MIN_INLIER_CONES})"
            )
            return None

        # initial row assignment: split by perpendicular offset from the cone
        # field's principal direction (robust even if the rows are tilted in
        # the local frame and cross y=0)
        centered = candidates - candidates.mean(axis=0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        row_normal = np.array([-vt[0, 1], vt[0, 0]])
        initial_is_left = centered @ row_normal > 0.0

        fit = fit_two_parallel_rows(candidates, initial_is_left=initial_is_left)
        if fit is None:
            self._last_reject = (
                f"rejected: degenerate fit on {len(candidates)} candidates"
            )
            return None
        slope, intercept_left, intercept_right, is_left, inlier_mask = fit

        inlier_cones = candidates[inlier_mask]
        span = float(np.ptp(inlier_cones[:, 0])) if len(inlier_cones) else 0.0
        n_left = int((inlier_mask & is_left).sum())
        n_right = int((inlier_mask & ~is_left).sum())
        stats = (
            f"slope={np.rad2deg(np.arctan(slope)):+.2f}deg(local) "
            f"inliers={len(inlier_cones)}/{len(candidates)} (L{n_left}/R{n_right}) "
            f"span={span:.1f}m yaw={np.rad2deg(vehicle_yaw):+.2f}deg"
        )

        if len(inlier_cones) < MIN_INLIER_CONES:
            self._last_reject = f"rejected: too few inliers | {stats}"
            return None
        # a short baseline cannot give a trustworthy direction
        if span < MIN_INLIER_SPAN_M:
            self._last_reject = f"rejected: span<{MIN_INLIER_SPAN_M}m | {stats}"
            return None
        # the car drives along the lane; a fit far from the current heading is
        # a bad fit, not a real track direction
        if np.abs(np.arctan(slope)) > MAX_ANGLE_TO_HEADING_RAD:
            self._last_reject = f"rejected: fit angle over limit | {stats}"
            return None

        # lane centerline offset from the car (local frame); only trustworthy
        # when both rows are actually seen
        center_intercept = None
        if (
            n_left >= MIN_ROW_INLIERS_FOR_LATERAL
            and n_right >= MIN_ROW_INLIERS_FOR_LATERAL
            and MIN_LANE_WIDTH_M < (intercept_left - intercept_right) < MAX_LANE_WIDTH_M
        ):
            center_intercept = (intercept_left + intercept_right) / 2.0

        return {
            "dir_map": float(np.arctan(slope) + vehicle_yaw),
            "center_intercept": center_intercept,
            "span": span,
            "stats": stats,
            "all_cones": all_cones,
            "candidates": candidates,
            "inlier_mask": inlier_mask,
            "vehicle_yaw": vehicle_yaw,
        }

    def do_relocalization_once(
        self,
        cones: list[FloatArray],
        vehicle_position: FloatArray,
        vehicle_direction: FloatArray,
    ) -> tuple[RelocalizationCallable, RelocalizationCallable] | None:
        # Accept immediately with the path running straight from the car's
        # current pose along its heading — the fit does NOT set the start
        # vector, so initial steering is ~0. Gentle trimming toward the
        # measured rows only starts afterwards, in _track.
        self._theta = float(np.arctan2(*vehicle_direction[::-1]))
        self._initial_theta = self._theta
        self._anchor = np.asarray(vehicle_position, dtype=float).copy()
        self._lateral = 0.0
        self._tracked_span = 0.0

        # run the fit for the log so the initial disagreement is visible
        fit = self._fit_rows(cones, vehicle_position, vehicle_direction)
        if fit is not None:
            center = fit["center_intercept"]
            center_str = "n/a" if center is None else f"{center:+.2f}m"
            initial_fit = (
                f"initial fit (not applied): dir={np.rad2deg(fit['dir_map']):+.2f}deg(map) "
                f"center={center_str} | {fit['stats']}"
            )
        else:
            initial_fit = f"initial fit (not applied): {self._last_reject}"
        self.last_report = (
            f"ACCEPTED (straight start): heading "
            f"{np.rad2deg(self._theta):+.2f}deg(map) from staging point, "
            f"gentle trim active (max {np.rad2deg(MAX_TOTAL_DIRECTION_RAD):.0f}deg/"
            f"{MAX_TOTAL_LATERAL_M:.1f}m total) | {initial_fit}"
        )

        return self._build_transforms()

    def _track(
        self,
        cones: list[FloatArray],
        vehicle_position: FloatArray,
        vehicle_direction: FloatArray,
    ) -> None:
        fit = self._fit_rows(cones, vehicle_position, vehicle_direction)
        if fit is None:
            return
        # ignore fits that see much less track than we already had
        if fit["span"] < max(
            MIN_INLIER_SPAN_M, self._tracked_span - MAX_SPAN_REGRESSION_M
        ):
            return

        car_known, _ = self.transform_to_known_map_frame(vehicle_position, 0.0)

        # direction: bounded step, rotating about the path point at the car so
        # the heading correction does not move the path sideways under the car
        delta_dir = (fit["dir_map"] - self._theta + np.pi) % (2 * np.pi) - np.pi
        # a fit that disagrees strongly with the current path is a bad fit,
        # not a missteering — do not chase it
        if np.abs(delta_dir) > MAX_FIT_DISAGREEMENT_RAD:
            return
        step_dir = float(
            np.clip(delta_dir, -MAX_DIRECTION_STEP_RAD, MAX_DIRECTION_STEP_RAD)
        )
        # hard cap: the accumulated correction never exceeds
        # MAX_TOTAL_DIRECTION_RAD away from the initial straight heading
        new_theta = self._theta + step_dir
        total_dir = (new_theta - self._initial_theta + np.pi) % (2 * np.pi) - np.pi
        total_dir = float(
            np.clip(total_dir, -MAX_TOTAL_DIRECTION_RAD, MAX_TOTAL_DIRECTION_RAD)
        )
        new_theta = self._initial_theta + total_dir

        path_point_at_car = np.array([car_known[0], 0.0])
        path_point_map, _ = self.transform_to_original_frame(path_point_at_car, 0.0)
        self._theta = new_theta
        self._anchor = path_point_map - rotate(
            path_point_at_car + np.array([0.0, self._lateral]), self._theta
        )

        # lateral: bounded step onto the measured lane centerline
        if fit["center_intercept"] is not None:
            lateral_error = fit["center_intercept"] + car_known[1]
            step_lat = float(
                np.clip(lateral_error, -MAX_LATERAL_STEP_M, MAX_LATERAL_STEP_M)
            )
            self._lateral = float(
                np.clip(
                    self._lateral + step_lat,
                    -MAX_TOTAL_LATERAL_M,
                    MAX_TOTAL_LATERAL_M,
                )
            )

        self._tracked_span = max(self._tracked_span, fit["span"])
        (
            self._original_to_known_map_transformer,
            self._known_map_to_original_transformer,
        ) = self._build_transforms()

        # log only when the tracked state moves meaningfully
        bucket = (
            round(np.rad2deg(self._theta), 1),
            round(self._lateral, 1),
            int(self._tracked_span // 5),
        )
        if bucket != self._last_report_bucket:
            self._last_report_bucket = bucket
            self.last_report = (
                f"TRACKING: dir={np.rad2deg(self._theta):+.2f}deg "
                f"lateral={self._lateral:+.2f}m span={self._tracked_span:.1f}m "
                f"| {fit['stats']}"
            )

    def get_known_global_path(self) -> FloatArray:
        return BASE_ACCELERATION_PATH


def create_acceleartion_path() -> FloatArray:
    # A single straight forward leg. The path must NOT contain a return leg:
    # the local-path extraction picks global-path points by euclidean distance
    # to the car, so a parallel return leg a few metres to the side gets
    # blended into the served path and pulls it sideways (or, with enough
    # lateral offset, the extraction locks onto it entirely and the path
    # U-turns).
    path_x = np.arange(-10, 150, 0.2)
    rng = np.random.default_rng(42)
    # add a bit of noise so that path calculations do not crash
    path_y = rng.normal(0, 0.01, len(path_x))

    return np.array([path_x, path_y]).T


BASE_ACCELERATION_PATH = create_acceleartion_path()

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plt.scatter(
        *create_acceleartion_path().T, c=np.arange(len(create_acceleartion_path()))
    )

    plt.axis("equal")
    plt.show()
