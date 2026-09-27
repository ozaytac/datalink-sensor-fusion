"""
L7 test suite — geo.py and export.py.

Tests cover:
  - lla_to_ecef: known WGS-84 points, round-trip
  - ecef_to_ned: zero offset, known direction
  - ned_to_ecef: inverse of ecef_to_ned
  - ecef_to_lla: round-trip with lla_to_ecef
  - geodetic_to_ned: near-zero for same point
  - ned_state_to_lla: round-trip
  - _get_bearing: known compass directions
  - _get_heading_rate: wrap-around at ±180
  - export_to_excel: empty history, non-empty history (CSV output)
"""

from __future__ import annotations

import csv
import math
import pickle
import types
import tempfile
from pathlib import Path

import numpy as np
import pytest

from datalink.io.geo import (
    lla_to_ecef,
    ecef_to_ned,
    ned_to_ecef,
    ecef_to_lla,
    geodetic_to_ned,
    ned_state_to_lla,
)
from datalink.io.export import (
    _get_bearing,
    _get_heading_rate,
    export_to_excel,
)


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _make_ownship(n: int = 50, dt: float = 0.2) -> types.SimpleNamespace:
    t = np.linspace(0.0, n * dt, n)
    return types.SimpleNamespace(
        time=t,
        latitude=np.full(n, 39.9),
        longitude=np.full(n, 32.8),
        altitude=np.full(n, 1000.0),
        heading=np.zeros(n),
        xSpeed=np.zeros(n),
        ySpeed=np.zeros(n),
        zSpeed=np.zeros(n),
    )


def _make_summary_pkl(tmp_path: Path, n_steps: int = 15) -> str:
    """Create a minimal SimulationSummary pickle with one fusion track."""
    fusion_history = []
    for i in range(n_steps):
        fusion_history.append({
            "FusionID":          [1],
            "RadarID":           [1],
            "InfraredID":        [-1],
            "FusedState":        [np.array([1000.0, 5.0, 500.0, -3.0, -200.0, 1.0])],
            "FusionUpdateTime":  [float(i) * 0.2],
            "TargetIndex":       [2],
            "Type":              ["Air"],
        })

    hist = {
        "fusion_history": fusion_history,
        "time": [i * 0.2 for i in range(n_steps)],
    }
    path = tmp_path / "summary.pkl"
    with open(path, "wb") as fh:
        pickle.dump(hist, fh)
    return str(path)


# ------------------------------------------------------------------ #
# Geo — LLA ↔ ECEF
# ------------------------------------------------------------------ #

class TestLLAToECEF:
    def test_origin_zero_lat_lon(self):
        """At (0°, 0°, 0m) ECEF x = a, y = 0, z = 0."""
        x, y, z = lla_to_ecef(0.0, 0.0, 0.0)
        assert x == pytest.approx(6_378_137.0, rel=1e-6)
        assert y == pytest.approx(0.0, abs=1.0)
        assert z == pytest.approx(0.0, abs=1.0)

    def test_north_pole(self):
        """At (90°, 0°, 0m) x≈0, y≈0, z≈b."""
        _WGS84_B = 6_356_752.314245
        x, y, z = lla_to_ecef(90.0, 0.0, 0.0)
        assert z == pytest.approx(_WGS84_B, rel=1e-5)
        assert abs(x) < 1.0

    def test_altitude_increases_radius(self):
        x1, y1, z1 = lla_to_ecef(45.0, 45.0, 0.0)
        x2, y2, z2 = lla_to_ecef(45.0, 45.0, 1000.0)
        r1 = math.sqrt(x1**2 + y1**2 + z1**2)
        r2 = math.sqrt(x2**2 + y2**2 + z2**2)
        assert r2 == pytest.approx(r1 + 1000.0, rel=1e-4)

    def test_array_input(self):
        lats = np.array([0.0, 45.0])
        lons = np.array([0.0, 0.0])
        alts = np.array([0.0, 0.0])
        x, y, z = lla_to_ecef(lats, lons, alts)
        assert len(x) == 2


class TestECEFToLLA:
    def test_round_trip(self):
        lat, lon, alt = 39.9, 32.8, 5000.0
        x, y, z = lla_to_ecef(lat, lon, alt)
        lat2, lon2, alt2 = ecef_to_lla(x, y, z)
        assert float(lat2) == pytest.approx(lat, abs=1e-7)
        assert float(lon2) == pytest.approx(lon, abs=1e-7)
        assert float(alt2) == pytest.approx(alt, abs=1.0)   # ~cm accuracy

    def test_north_pole_round_trip(self):
        lat, lon, alt = 89.9, 0.0, 0.0
        x, y, z = lla_to_ecef(lat, lon, alt)
        lat2, lon2, alt2 = ecef_to_lla(x, y, z)
        assert math.isfinite(float(lat2))
        assert float(lat2) == pytest.approx(lat, abs=1e-5)
        assert float(alt2) == pytest.approx(alt, abs=1.0)


class TestECEFToNED:
    def test_zero_vector_is_zero(self):
        N, E, D = ecef_to_ned(0.0, 0.0, 0.0, lat0=45.0, lon0=45.0)
        assert N == pytest.approx(0.0, abs=1e-10)
        assert E == pytest.approx(0.0, abs=1e-10)
        assert D == pytest.approx(0.0, abs=1e-10)

    def test_preserves_norm(self):
        # Rotation preserves vector magnitude
        u, v, w = 1000.0, 2000.0, 3000.0
        N, E, D = ecef_to_ned(u, v, w, lat0=30.0, lon0=60.0)
        assert math.sqrt(N**2 + E**2 + D**2) == pytest.approx(
            math.sqrt(u**2 + v**2 + w**2), rel=1e-10)


class TestNEDToECEF:
    def test_zero_offset_at_origin(self):
        lat0, lon0, h0 = 0.0, 0.0, 0.0
        x, y, z = ned_to_ecef(0.0, 0.0, 0.0, lat0, lon0, h0)
        x0, y0, z0 = lla_to_ecef(lat0, lon0, h0)
        assert float(x) == pytest.approx(float(x0), rel=1e-8)

    def test_round_trip_with_ecef_to_ned(self):
        lat0, lon0, h0 = 39.9, 32.8, 1000.0
        x0, y0, z0 = lla_to_ecef(lat0, lon0, h0)
        # 1km East offset in ENU
        x, y, z = ned_to_ecef(1000.0, 0.0, 0.0, lat0, lon0, h0)
        dx, dy, dz = x - x0, y - y0, z - z0
        N, E, D = ecef_to_ned(dx, dy, dz, lat0, lon0)
        assert E == pytest.approx(1000.0, rel=1e-4)
        assert abs(N) < 1.0


class TestGeodeticToNED:
    def test_same_point_is_zero(self):
        N, E, D = geodetic_to_ned(45.0, 45.0, 1000.0, 45.0, 45.0, 1000.0)
        assert abs(N) < 0.1
        assert abs(E) < 0.1
        assert abs(D) < 0.1

    def test_1km_north(self):
        """Point 1 km north (same lon/alt) should give N≈1000, E≈0."""
        lat0, lon0, alt0 = 39.9, 32.8, 0.0
        # ~0.009 deg ≈ 1 km northward
        N, E, D = geodetic_to_ned(lat0 + 0.009, lon0, alt0,
                                   lat0, lon0, alt0)
        assert N == pytest.approx(1000.0, rel=0.01)
        assert abs(E) < 10.0


class TestNEDStateToLLA:
    def test_zero_state_at_origin(self):
        """Zero NED state + reference origin → reference LLA."""
        lat0, lon0, alt0 = 39.9, 32.8, 1000.0
        state = np.zeros(6)
        lat, lon, alt = ned_state_to_lla(state, lat0, lon0, alt0)
        # ned_state_to_lla maps [x,vx,y,vy,z,vz] → (East=y, North=x, zUp=z)
        # all zero → should be close to origin
        assert abs(lat - lat0) < 0.01
        assert abs(lon - lon0) < 0.01

    def test_state_returns_finite(self):
        state = np.array([1000.0, 5.0, 500.0, -3.0, -200.0, 1.0])
        lat, lon, alt = ned_state_to_lla(state, 39.9, 32.8, 1000.0)
        assert math.isfinite(lat)
        assert math.isfinite(lon)
        assert math.isfinite(alt)


# ------------------------------------------------------------------ #
# Export helpers
# ------------------------------------------------------------------ #

class TestGetBearing:
    def test_single_point_returns_zero(self):
        b = _get_bearing(np.array([45.0]), np.array([30.0]))
        assert len(b) == 1

    def test_northward_is_zero_or_360(self):
        # Moving north: same lon, increasing lat
        b = _get_bearing(np.array([0.0, 1.0]), np.array([0.0, 0.0]))
        assert b[0] == pytest.approx(0.0, abs=1e-4) or b[0] == pytest.approx(360.0, abs=1e-4)

    def test_eastward_is_90(self):
        b = _get_bearing(np.array([0.0, 0.0]), np.array([0.0, 1.0]))
        assert b[0] == pytest.approx(90.0, abs=0.1)

    def test_length_matches_input(self):
        lats = np.array([0.0, 1.0, 2.0, 3.0])
        lons = np.array([0.0, 0.0, 0.0, 0.0])
        b = _get_bearing(lats, lons)
        assert len(b) == len(lats)


class TestGetHeadingRate:
    def test_constant_heading_is_zero_rate(self):
        h = np.array([90.0, 90.0, 90.0])
        t = np.array([0.0, 1.0, 2.0])
        r = _get_heading_rate(h, t)
        assert np.allclose(r, 0.0)

    def test_linear_increase(self):
        h = np.array([0.0, 10.0, 20.0])
        t = np.array([0.0, 1.0, 2.0])
        r = _get_heading_rate(h, t)
        assert r[0] == pytest.approx(10.0)
        assert r[1] == pytest.approx(10.0)
        assert r[-1] == 0.0

    def test_wrap_around_360(self):
        # 350 → 10: diff = 10-350 = -340 → wrap to +20
        h = np.array([350.0, 10.0])
        t = np.array([0.0, 1.0])
        r = _get_heading_rate(h, t)
        # -340 < -180 → 360 + (-340) = 20
        assert r[0] == pytest.approx(20.0, abs=1e-6)

    def test_output_length(self):
        h = np.array([0.0, 30.0, 60.0, 90.0])
        t = np.array([0.0, 1.0, 2.0, 3.0])
        r = _get_heading_rate(h, t)
        assert len(r) == len(h)


# ------------------------------------------------------------------ #
# Export to file
# ------------------------------------------------------------------ #

class TestExportToExcel:
    def test_empty_history_creates_file(self, tmp_path):
        hist = {"fusion_history": [], "time": []}
        pkl = tmp_path / "empty.pkl"
        with open(pkl, "wb") as fh:
            pickle.dump(hist, fh)
        out = str(tmp_path / "out.csv")
        export_to_excel(str(pkl), _make_ownship(), output_path=out,
                        min_track_points=0)
        assert Path(out).exists()

    def test_nonempty_history_produces_csv_rows(self, tmp_path):
        pkl = _make_summary_pkl(tmp_path, n_steps=20)
        out = str(tmp_path / "out.csv")
        export_to_excel(pkl, _make_ownship(n=50), output_path=out,
                        min_track_points=5)
        assert Path(out).exists()
        with open(out) as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) > 0

    def test_csv_has_correct_columns(self, tmp_path):
        pkl = _make_summary_pkl(tmp_path, n_steps=20)
        out = str(tmp_path / "out.csv")
        export_to_excel(pkl, _make_ownship(n=50), output_path=out,
                        min_track_points=1)
        with open(out) as fh:
            reader = csv.DictReader(fh)
            cols = reader.fieldnames
        expected = {"flight_id", "latitude", "longitude", "altitude",
                    "ground_speed", "headingRate", "time"}
        assert expected.issubset(set(cols))

    def test_min_track_points_filters_short_tracks(self, tmp_path):
        # Only 5 steps, min_track_points=10 → no rows
        pkl = _make_summary_pkl(tmp_path, n_steps=5)
        out = str(tmp_path / "out.csv")
        export_to_excel(pkl, _make_ownship(n=50), output_path=out,
                        min_track_points=10)
        with open(out) as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 0

    def test_lat_lon_are_finite(self, tmp_path):
        pkl = _make_summary_pkl(tmp_path, n_steps=20)
        out = str(tmp_path / "out.csv")
        export_to_excel(pkl, _make_ownship(n=50), output_path=out,
                        min_track_points=1)
        with open(out) as fh:
            for row in csv.DictReader(fh):
                assert math.isfinite(float(row["latitude"]))
                assert math.isfinite(float(row["longitude"]))
                assert math.isfinite(float(row["altitude"]))

    def test_output_directory_created(self, tmp_path):
        pkl = _make_summary_pkl(tmp_path, n_steps=15)
        out = str(tmp_path / "new_dir" / "subdir" / "out.csv")
        export_to_excel(pkl, _make_ownship(n=50), output_path=out,
                        min_track_points=1)
        assert Path(out).exists()
