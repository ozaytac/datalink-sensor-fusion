# DataLink_v2.0 Python Port

**DataLink_v2.0** is a pure Python/NumPy/SciPy multi-sensor (radar + infrared) target tracking and track-to-track fusion simulation, with zero external toolbox dependencies.

> See the full **[Interface Control Document](docs/ICD.md)** for the complete interface specification.

## What this is

`datalink` simulates one or more "ownship" platforms, each carrying a radar sensor (GNN tracker + EKF), an infrared sensor (MSC-EKF), and a fusion manager that combines radar (Cartesian) and IR (angle-only, Modified Spherical Coordinates) tracks via LMMSE track-to-track fusion. It can stream results as UDP telemetry, record full simulation history to disk, compute OSPA/Hausdorff tracking-accuracy metrics, and export fused tracks to Excel/CSV.

### Layered architecture

| Layer | Contents |
| --- | --- |
| L1 | `transformations/` (MSC ↔ Cartesian, Jacobians) |
| L2 | `sensors/radar/` (sensor + tracking models), `sensors/infrared/` (MSC-EKF, IR sensor stub) |
| L3 | `core/ownship.py` (platform orchestration) |
| L4 | `core/simulation_summary.py`, `fusion/` (track-to-track fusion) |
| L5 | `core/simulation.py` (main loop), `core/udp_functions.py` (telemetry) |
| L6 | `error_metrics/` (OSPA, Hausdorff, fusion error) |
| L7 | `io/geo.py` (WGS-84 conversions), `io/export.py` (Excel/CSV) |

## Installation

```bash
pip install numpy scipy scikit-learn pandas openpyxl pymap3d
pip install pytest   # for running the test suite
```

## Package layout

```
py_datalink/
├── datalink/
│   ├── __init__.py                  → Ownship, SimulationSummary, FusionManager
│   │
│   ├── core/                        → platform + simulation orchestration
│   │   ├── ownship.py                → Ownship platform class (L3)
│   │   ├── simulation_summary.py     → per-step history recorder (L4)
│   │   ├── simulation.py             → main simulation loop (L5)
│   │   └── udp_functions.py          → UDP telemetry senders (L5)
│   │
│   ├── io/                          → file/coordinate I/O (L7)
│   │   ├── geo.py                    → WGS-84 coordinate conversions
│   │   └── export.py                 → Excel/CSV export
│   │
│   ├── sensors/                     → sensor models (L2)
│   │   ├── radar/                    → radar subsystem
│   │   │   ├── detection.py           → Detection data class
│   │   │   ├── track.py               → Track data class
│   │   │   ├── fluctuation_models.py  → Swerling RCS models
│   │   │   ├── radar_beam.py          → beam pointing management
│   │   │   ├── snjr.py                → SNJR equation
│   │   │   ├── target.py              → Target object
│   │   │   ├── clutter.py             → ground clutter model
│   │   │   ├── mpar.py                → MPAR sensor
│   │   │   ├── radar_sensor.py        → top-level RadarSensor
│   │   │   ├── simulation_history.py  → simulation history helper
│   │   │   ├── tracking_radar.py      → single-function tracking radar
│   │   │   ├── jammers/jammer.py      → jammer model
│   │   │   └── tracking/              → EKF, GNN Tracker, DBSCAN
│   │   │       ├── cvekf.py            → constant-velocity EKF
│   │   │       ├── gnn_tracker.py      → GNN Tracker (Hungarian assignment)
│   │   │       ├── cost.py             → assignment cost matrix
│   │   │       ├── dbscan_filter.py    → DBSCAN clustering filter
│   │   │       └── constvel.py         → constant-velocity motion model
│   │   │
│   │   └── infrared/                 → IR subsystem
│   │       ├── mscekf.py              → MSC-EKF filter
│   │       ├── init_filters.py        → init_cv_msc_ekf, init_msc_rp_ekf
│   │       └── infrared_sensor.py     → InfraredSensor (angle-only stub)
│   │
│   ├── transformations/             → coordinate transformations (L1)
│   │   ├── msc.py                    → MSC ↔ Cartesian
│   │   └── gradients.py              → Jacobian matrices
│   │
│   ├── fusion/                      → fusion manager (L4)
│   │   └── fusion_manager.py         → FusionManager, FusionEntry
│   │
│   └── error_metrics/               → error metrics (L6)
│       ├── ospa.py                   → OSPA metric (Cartesian + MSC)
│       ├── hausdorff.py              → Hausdorff distance
│       └── optimization_errors.py    → fusion position/angle errors
│
├── docs/
│   ├── ICD.md                       → Interface Control Document
│
└── tests/                           → 178 automated tests
    ├── test_transformations.py
    ├── test_ir_filters.py
    ├── test_radar.py
    ├── test_fusion.py
    ├── test_simulation.py
    ├── test_error_metrics.py
    └── test_export.py
```

## Quick start

### 1. Coordinate transformation

```python
from datalink.transformations import cartesian_to_msc, msc_to_cartesian
import numpy as np

# Cartesian state vector: [x, vx, y, vy, z, vz]
state_cart = np.array([1000.0, 50.0, 500.0, -20.0, -200.0, 10.0])

# Convert to MSC: [az, omega, el, elDot, invR, dRinvR]  (radians)
state_msc = cartesian_to_msc(state_cart)

# Convert back
state_back = msc_to_cartesian(state_msc)
```

### 2. MSC-EKF (IR filter)

```python
from datalink.sensors.infrared import init_cv_msc_ekf
import numpy as np

# init_cv_msc_ekf takes a plain dict (NOT a Detection object), angles in DEGREES
detection = {
    "measurement": [30.0, 18.0],                 # [az_deg, el_deg]
    "measurement_noise": np.diag([1.0, 1.0]),     # degrees^2
}
ekf = init_cv_msc_ekf(detection)

# Prediction step
ekf.predict(dt=0.2)

# Update step measurement is [az_rad, el_rad], a 2-vector (NOT 4 elements)
innovation = ekf.update(np.array([0.5240, 0.3145]))

print("State:", ekf.state)
print("Covariance:\n", ekf.covariance)
```

### 3. GNN Tracker (radar tracking)

```python
from datalink.sensors.radar.tracking.gnn_tracker import GNNTracker
from datalink.sensors.radar.detection import Detection
import numpy as np

tracker = GNNTracker(
    confirm_threshold=(2, 3),   # (hits, steps) to confirm
    delete_threshold=(3, 4),    # (misses, steps) to delete
)

for step in range(20):
    t = step * 0.2
    meas = np.array([1000.0 + step * 5, 500.0, -200.0])
    det = Detection(
        time=t,
        measurement=meas,
        measurement_noise=np.eye(3) * 100,
    )

    confirmed_tracks, all_tracks = tracker.step([det], sim_time=t, dt=0.2)
    print(f"Step {step}: {len(confirmed_tracks)} confirmed track(s)")
```

### 4. Fusion Manager

```python
from datalink.fusion.fusion_manager import FusionManager
from datalink.sensors.radar.track import Track
import numpy as np

params = {
    "scenarioParams": {
        "HasRadar": True, "HasInfrared": False, "HasEW": False,
        "FusionManager": "FusionManager",
        "updateRate": 5, "trajFileUpdateRate": 50, "DataLink": False,
    },
}

fm = FusionManager(traj=None, params=params)

radar_track = Track(
    track_id=1,
    state=np.array([1000.0, 10.0, 500.0, -5.0, -200.0, 3.0]),
    state_covariance=np.eye(6) * 1e4,
    update_time=1.0,
    object_attributes={"TargetIndex": 1},
)

fm.update_manager([radar_track], [], time=1.0)
fm.update_manager([radar_track], [], time=1.2)

for t in fm.get_fused_tracks():
    print(f"Fusion ID: {t['fusion_id']}, state: {t['state']}")
```

### 5. Ownship (full pipeline)

```python
import types, numpy as np
from datalink.core.ownship import Ownship

traj = types.SimpleNamespace(
    Position=np.zeros(3),
    Trajectory=types.SimpleNamespace(Velocity=np.zeros(3), Acceleration=np.zeros(3)),
)

params = {
    "scenarioParams": {
        "HasRadar": False, "HasInfrared": False, "HasEW": False,
        "FusionManager": "FusionManager",
        "updateRate": 5, "trajFileUpdateRate": 50, "DataLink": False,
    },
    "trackingRadar": {"PRF": 1000, "theta_FOV": 60, "phi_FOV": 30},
    "IRParams": {},
}

ownship = Ownship(
    platform_id=1, trajectory=traj, params=params,
    blue1_idx=1, initial_pos=np.zeros(3), initial_lla=np.zeros(3),
    num_ownship=1, num_target=0,
)

fused_tracks = ownship.update_fusion(time=1.0, confirmed_radar_tracks=[], confirmed_ir_tracks=[])
```

> Note: `trackingRadar`/`clutterParams`/`trackerParams` above are only sufficient when `HasRadar=False`. See [docs/ICD.md](docs/ICD.md) §2.2.1 for the full field list required when radar is enabled.

### 6. Simulation loop

```python
from datalink.core.simulation import Simulation

sim = Simulation("data/params.json", enable_udp=False)
history = sim.run()
print(f"Ran {len(history)} steps.")
```

Minimal `params.json`:

```jsonc
{
  "scenarioParams": {
    "HasRadar": true,
    "HasInfrared": false,
    "HasEW": false,
    "FusionManager": "FusionManager",
    "updateRate": 5,
    "trajFileUpdateRate": 50,
    "DataLink": false
  },
  "trackingRadar": {"PRF": 1000, "theta_FOV": 60, "phi_FOV": 30},
  "IRParams": {}
}
```

See [docs/ICD.md](docs/ICD.md) for the complete `params.json` schema (required when `HasRadar: true`).

### 7. UDP telemetry

```python
from datalink.core.udp_functions import UDPSender, get_telemetry_tracks, get_telemetry_fusion_tracks

# port 5001 = radar, 5002 = IR, 5003 = fusion
with UDPSender(port=5001, host="127.0.0.1") as sender:
    radar_msg, ir_msg = get_telemetry_tracks(
        ownship_id=1,
        radar_tracks=confirmed_radar_tracks,
        ir_tracks=confirmed_ir_tracks,
        time=1.5,
    )
    sender.send(radar_msg)
```

See [docs/ICD.md](docs/ICD.md) §2.1 for the full wire-format field tables (radar/IR/fusion messages).

### 8. OSPA metric

```python
from datalink.error_metrics import ospa, ospa_cartesian, ospa_msc
import numpy as np

X = np.array([[1000.0, 500.0, -200.0]])   # estimate
Y = np.array([[1005.0, 498.0, -201.0]])   # ground truth
total, loc, card, label = ospa(X, Y, c=100.0, p=2.0)
print(f"OSPA={total:.2f}m  loc={loc:.2f}  card={card:.2f}")

# Track objects (Cartesian)
total, loc, card, _ = ospa_cartesian(confirmed_radar_tracks, true_tracks, c=100.0)

# Track objects (MSC angular, for IR)
total, loc, card, _ = ospa_msc(confirmed_ir_tracks, true_tracks, c=0.1)  # c in radians
```

### 9. Geodetic conversions (WGS-84)

```python
from datalink.io.geo import lla_to_ecef, ecef_to_lla, geodetic_to_ned, ned_state_to_lla

x, y, z = lla_to_ecef(lat=39.9, lon=32.8, alt=1000.0)
lat, lon, alt = ecef_to_lla(x, y, z)

N, E, D = geodetic_to_ned(
    lat=39.91, lon=32.81, alt=1200.0,
    lat0=39.9,  lon0=32.8,  alt0=1000.0,
)
print(f"North={N:.1f}m  East={E:.1f}m  Down={D:.1f}m")

ned_state = np.array([1000.0, 5.0, 500.0, -3.0, -200.0, 1.0])
tgt_lat, tgt_lon, tgt_alt = ned_state_to_lla(ned_state, lat0=39.9, lon0=32.8, alt0=1000.0)
```

### 10. Excel/CSV export

```python
from datalink.io.export import export_to_excel
import types, numpy as np

ownship_data = types.SimpleNamespace(
    time=sim_times,
    latitude=lat_array, longitude=lon_array, altitude=alt_array,
    heading=hdg_array,
    xSpeed=vx_array, ySpeed=vy_array, zSpeed=vz_array,
)

export_to_excel(
    summary_path="outputs/summary_BLUE11_unknown_unknown.pkl",
    ownship_data=ownship_data,
    output_path="SenaryoCSV/scene1.xlsx",
    min_track_points=10,
)
# Falls back to .csv automatically if pandas is not installed.
```

## Running the tests

```bash
python -m pytest tests/ -v

# by layer
python -m pytest tests/test_transformations.py -v   # L1
python -m pytest tests/test_radar.py -v              # L2
python -m pytest tests/test_fusion.py -v              # L3/L4
python -m pytest tests/test_simulation.py -v          # L5
python -m pytest tests/test_error_metrics.py -v       # L6
python -m pytest tests/test_export.py -v              # L7
```

## Coordinate system conventions

- **Cartesian CV state vector**: `[x, vx, y, vy, z, vz]` NED metres; position at indices `[0,2,4]`, velocity at `[1,3,5]`.
- **MSC state vector**: `[az, omega, el, elDot, invR, dRinvR]` radians / (1/metres).
- **`datalink.io.geo`** uses degrees throughout; **`datalink.transformations`**/tracking modules use radians throughout. See [docs/ICD.md](docs/ICD.md) §4.3 for the full unit-boundary discussion.
- **Swerling models**: `'Swerling0'`/`'Swerling5'` → constant RCS; `'Swerling1'`/`'Swerling2'` → `Gamma(a=1, scale=σ)`; `'Swerling3'`/`'Swerling4'` → `Gamma(a=2, scale=σ/2)`.

## Interface Control Document

For the full external/internal interface specification (UDP wire formats, the complete `params.json` schema, `.pkl`/Excel output schemas, and every module's public API), see **[docs/ICD.md](docs/ICD.md)**.

## License

MIT. See [LICENSE](LICENSE).
