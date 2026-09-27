# Interface Control Document (ICD)

**Document title:** DataLink_v2.0 (Python Port) — Interface Control Document
**Version:** 1.0
**Date:** 2026-09-27
**Status:** Draft — generated from source-code inspection of the `datalink` package
**Applies to:** `py_datalink` repository, package `datalink`

---

## 1. Introduction

### 1.1 Purpose

This document specifies the external and internal interfaces of the `datalink` Python package — a pure Python/NumPy/SciPy implementation of **DataLink_v2.0**, a multi-sensor target tracking and track-to-track fusion simulation. It is intended to let an external system, or a developer extending the package, implement a compatible client or module without reading the full source tree.

### 1.2 Scope

This ICD covers:

- **External interfaces**: the UDP telemetry wire protocol, the `params.json` configuration file schema, the `.pkl` simulation output formats, the Excel/CSV export output, and the trajectory input structure.
- **Internal interfaces**: the public data contracts (function signatures, class constructors, return shapes) between the major modules of the package — the level of detail needed to call one module's output as another's input correctly.

It does **not** cover implementation details of numerical algorithms (radar/IR physics models, EKF math) beyond what is needed to use their inputs/outputs correctly.

### 1.3 Referenced Documents

- `README.md` — project overview and quick start

### 1.4 Definitions, Acronyms, Abbreviations

| Term | Meaning |
|---|---|
| MSC | Modified Spherical Coordinates — angle/angle-rate/inverse-range state representation |
| EKF | Extended Kalman Filter |
| GNN | Global Nearest Neighbour (multi-target data association) |
| OSPA | Optimal Subpattern Assignment — multi-object tracking accuracy metric |
| NED | North-East-Down local Cartesian frame |
| ECEF | Earth-Centered, Earth-Fixed Cartesian frame |
| LLA | Latitude / Longitude / Altitude (geodetic) |
| WGS-84 | World Geodetic System 1984 ellipsoid |
| RCS | Radar Cross Section |
| PRF | Pulse Repetition Frequency |
| FOV | Field of View |
| DBSCAN | Density-Based Spatial Clustering of Applications with Noise |
| ILP | Integer Linear Programming (assignment problem, solved here via the Hungarian algorithm) |
| T2TF | Track-to-Track Fusion |
| LMMSE | Linear Minimum Mean Square Error (estimator) |

---

## 2. External Interfaces

### 2.1 UDP Telemetry Interface

Implemented in `datalink/udp_functions.py`. Transport class: `UDPSender` — one UDP datagram socket (`socket.AF_INET`, `socket.SOCK_DGRAM`) per port, unicast to a fixed host/port, default host `127.0.0.1`. Payload encoding: **UTF-8 text**, sent as a single `sendto()` call per message (no length-prefix, no framing — one message = one UDP datagram).

| Port | Content | Producer function |
|---|---|---|
| 5001 | Radar tracks | `get_telemetry_tracks()` (first return value) |
| 5002 | IR tracks | `get_telemetry_tracks()` (second return value) |
| 5003 | Fusion tracks | `get_telemetry_fusion_tracks()` |

`Simulation` (see §3.9) opens exactly these three ports when `enable_udp=True`. `udp_functions.make_senders()` additionally opens port 5000, but no message producer currently targets it — **reserved/unused**.

**Message framing:** each message string is a list of per-track *entries*, joined with `;` (semicolon). Each entry is a list of comma-separated fields, in the fixed order given below. There is no trailing delimiter and no message emitted if the entry list is empty (`_senders[port].send()` is only called when the formatted string is non-empty).

#### 2.1.1 Radar track message (port 5001)

Format string per entry: `ownship_id,R,FoF,trackId,time,X,Y,Z`

| Field | Type | Unit | Description |
|---|---|---|---|
| `ownship_id` | int | — | Platform ID of the reporting ownship (`Ownship.ID`) |
| `R` | literal `"R"` | — | Message-type tag (Radar) |
| `FoF` | int | — | Friend-or-foe flag, from `track.object_attributes['FoF']`, default `0` |
| `trackId` | int | — | Radar track ID (`Track.track_id`) |
| `time` | float, `%.6f` | s | Simulation time |
| `X` | float, `%.4f` | m | NED X (North) position, `state[0]` |
| `Y` | float, `%.4f` | m | NED Y (East) position, `state[2]` |
| `Z` | float, `%.4f` | m | NED Z (Down) position, `state[4]` |

Example (one track): `1,R,0,3,12.400000,1024.5000,-350.2000,-800.1000`

#### 2.1.2 IR track message (port 5002)

Format string per entry: `ownship_id,I,FoF,trackId,time,az_deg,el_deg`

| Field | Type | Unit | Description |
|---|---|---|---|
| `ownship_id` | int | — | Reporting ownship platform ID |
| `I` | literal `"I"` | — | Message-type tag (Infrared) |
| `FoF` | int | — | Friend-or-foe flag, default `0` |
| `trackId` | int | — | IR track ID |
| `time` | float, `%.6f` | s | Simulation time |
| `az_deg` | float, `%.6f` | deg | Azimuth, converted from MSC state `state[0]` (radians) |
| `el_deg` | float, `%.6f` | deg | Elevation, converted from MSC state `state[2]` (radians) |

Example: `1,I,0,7,12.400000,15.230000,-2.410000`

#### 2.1.3 Fusion track message (port 5003)

Format string per entry:
`ownship_id,F,FoF,fusionId,time,is_xyz,tX,tY,tZ,tAz,tEl,is_radar,rId,rX,rY,rZ,is_ir,irId,irAz,irEl`

| Field | Type | Unit | Description |
|---|---|---|---|
| `ownship_id` | int | — | Reporting ownship platform ID |
| `F` | literal `"F"` | — | Message-type tag (Fusion) |
| `FoF` | int/`0` | — | `fusion_track['FoF']`, default `0` (**not** cast to `int`; whatever type is stored is inserted as-is) |
| `fusionId` | int | — | `fusion_track['fusion_id']` |
| `time` | float, `%.6f` | s | Simulation time |
| `is_xyz` | 0/1 | — | 1 if a fused Cartesian state was available |
| `tX,tY,tZ` | float, `%.4f` | m | Fused NED position (`state[0],state[2],state[4]`); 0.0 if `is_xyz=0` |
| `tAz,tEl` | float, `%.6f` | deg | Fused bearing/elevation computed from `tX,tY,tZ` via `atan2`; 0.0 if `is_xyz=0` |
| `is_radar` | 0/1 | — | 1 if a radar sub-track contributed |
| `rId` | int | — | `fusion_track['radar_id']`, `0` if `is_radar=0` |
| `rX,rY,rZ` | float, `%.4f` | m | Radar sub-track NED position; `0` (int) if `is_radar=0` |
| `is_ir` | 0/1 | — | 1 if an IR sub-track contributed |
| `irId` | int | — | `fusion_track['ir_id']`, `0` if `is_ir=0` |
| `irAz,irEl` | float, `%.6f` | deg | IR sub-track azimuth/elevation, converted from MSC radians; `0` (int) if `is_ir=0` |

Example: `1,F,0,2,12.400000,1,1024.5000,-350.2000,-800.1000,341.0500,17.8000,1,3,1024.1000,-349.9000,-799.8000,0,0,0.000000,0.000000`

**Interface hazard:** radar/IR sub-fields fall back to `int 0` when absent, while the top-level `tX/tY/tZ` fall back to `float 0.0` — a receiving parser must not assume a fixed decimal-point format for every field; parse by splitting on `,` and casting per the table above, not by fixed column width.

### 2.2 File Interfaces

#### 2.2.1 `params.json` — scenario configuration

Loaded by `Simulation._load_params()` via plain `json.load()`. Top-level object with (at minimum) these sections:

**`scenarioParams`** (object, required)

| Field | Type | Required/Default | Description |
|---|---|---|---|
| `HasRadar` | bool | required for radar subsystem | Enables `RadarSensor` construction in `Ownship` |
| `HasInfrared` | bool | required for IR subsystem | Enables `InfraredSensor` construction |
| `HasEW` | bool | accepted but **not read anywhere in current code** | Reserved for a future electronic-warfare subsystem |
| `FusionManager` | str | default `'FusionManager'` | Read but not currently used to select an alternate implementation — `FusionManager` is always instantiated |
| `updateRate` | number (Hz) | required | Simulation step rate; `Simulation.dt = 1/updateRate` |
| `trajFileUpdateRate` | number (Hz) | default `50.0` | Trajectory file's native sample rate, used for coarse time interpolation in `Ownship._produce_target_relative_positions` |
| `DataLink` | bool | default falsy | Used only to build the output filename tag in `SimulationSummary.save()` |
| `num_steps` | int | default `100` | Only used when no trajectory file is supplied (`Simulation` builds a dummy stationary trajectory) |
| `doDBSCAN` | bool | default `False` | Enables DBSCAN detection clustering inside `MPAR` |
| `trajFile` | str | default `'unknown'` | Used only to build the output `.pkl` filename |
| `sensorNoise` | str | default `'unknown'` | Used only to build the output `.pkl` filename |

**`trackingRadar`** (object, required when `HasRadar=True`)

The minimal example in `KULLANIM_KILAVUZU.txt` (`{PRF, theta_FOV, phi_FOV}`) is **only sufficient when `HasRadar=False`** (i.e. `RadarSensor` is never constructed). If `HasRadar=True`, `TrackingRadar.__init__` requires the full set below (`KeyError` otherwise):

| Field | Type | Unit | Used by |
|---|---|---|---|
| `PRF` | number | Hz | `Ownship` (target-position interpolation), `TrackingRadar` |
| `theta_FOV` | number | deg | `RadarSensor`, `TrackingRadar` (azimuth half-FOV) |
| `phi_FOV` | number | deg | `RadarSensor`, `TrackingRadar` (elevation half-FOV) |
| `pulsewidth` | number | s | `TrackingRadar.bandwidth = 1/pulsewidth` |
| `theta_3db` | number | deg | Antenna beamwidth (azimuth) |
| `phi_3db` | number | deg | Antenna beamwidth (elevation) |
| `peak_power` | number | W | Radar equation |
| `wavelength` | number | m | Radar equation; `center_freq = c/wavelength` |
| `effective_noise_temperature` | number | K | Radar equation |
| `noise_figure` | number | dB | Radar equation |
| `false_alarm_probability` | number | — | Albersheim detectability |
| `detection_probability` | number | — | Albersheim detectability |
| `K_range`, `K_theta`, `K_phi` | number | — | Measurement-noise scale constants |
| `L_range`, `L_theta`, `L_phi` | number | — | Measurement-noise scale constants |
| `sidelobe_diff` | number | dB | Antenna sidelobe level |
| `range_bias`, `theta_bias`, `phi_bias` | number | m / deg | Deterministic measurement bias |
| `radiation_efficiency_factor` | number | — | Antenna gain calculation |
| `aperture_efficiency` | number | — | Antenna gain calculation |

**`clutterParams`** (object, required when `HasRadar=True`) — passed to `Clutter.__init__`:

| Field | Type | Unit | Description |
|---|---|---|---|
| `resolution_in_surface` | number | m | Surface clutter bin resolution |
| `range_limit` | number | m | Max range considered for clutter |
| `scatterer_distribution` | str | — | `'auto'` \| `'gaussian'` \| `'chisq_dof2'` \| `'chisq_dof4'` |
| `sigma_0` | number | dB·m²/m² | Clutter reflectivity coefficient |
| `tau` | number | s | Pulse width (clutter model) |
| `dbscan_epsilon` | number | m | Optional, default `500.0` — DBSCAN neighbourhood radius (only read when `doDBSCAN=True`) |

**`trackerParams`** (object, required when `HasRadar=True`) — passed to `GNNTracker` via `MPAR`:

| Field | Type | Default | Description |
|---|---|---|---|
| `max_tracks` | int | 100 | Maximum simultaneous tracks |
| `gate` | number (m) | 10000 | Assignment distance gate |
| `confirm_threshold` | `[int, int]` | `[2, 3]` | `(M hits, N steps)` to confirm a track |
| `delete_threshold` | `[int, int]` | `[3, 4]` | `(M misses, N steps)` to delete a track |
| `has_cost_matrix` | bool | `False` | Accept externally supplied cost matrix |

**`IRParams`** (object, required when `HasInfrared=True`) — read by `Ownship._initialize_sensors`, all optional with defaults:

| Field | Type | Default | Description |
|---|---|---|---|
| `LensDiameter` | number (m) | 0.08 | |
| `FocalLength` | number (m) | 800 | |
| `NumDetectors` | `[int, int]` | `[1000, 1000]` | Focal-plane array size |
| `CutoffFrequency` | number (Hz) | 20000 | |
| `DetectorArea` | number (m²) | 1.44e-6 | |
| `Detectivity` | number | 1.2e10 | D* [cm·Hz^0.5/W] |
| `NoiseEquivalentBandwidth` | number (Hz) | 30 | |
| `FalseAlarmRate` | number | 1e-6 | |
| `HasElevation` | bool | `True` | |
| `HasNoise` | bool | `True` | |
| `HasFalseAlarms` | bool | `True` | |
| `AzimuthBiasFraction` | number | 0.1 | |
| `ElevationBiasFraction` | number | 0.1 | |
| `ScanMode` | str | `'Mechanical'` | |
| `AzimuthScanLimits` | `[number, number]` (deg) | `[-60, 60]` | Used to split the IR FOV into scan cells |

`traj_path` (top-level, optional) — path to a trajectory file (`.mat`, `.npy`/`.npz`, or `.pkl`); overridden by the `traj_path` constructor argument to `Simulation` if given.

#### 2.2.2 `.pkl` — simulation history output

`Simulation._save_history()` writes two kinds of pickle files:

- `results/sim_history.pkl` — `pickle.dump(self.history)`, a `list[dict]`, one dict per timestep with keys `time` (float), `ownships` (`list[dict]` with keys `id`, `radar_tracks`, `ir_tracks`, `fused_tracks`).
- `outputs/summary_BLUE1<id>_<sensorNoise>_[w-DataLink_]<trajFile>[_<monte_carlo_idx>].pkl` — one per ownship, written by `SimulationSummary.save()` as `pickle.dump(self.__dict__)`. Top-level dict keys: `time` (`list[float]`), `radar_detections`, `ir_detections`, `ew_detections`, `radar_tracks`, `ir_tracks` (angular components pre-converted to **degrees**), `ew_tracks`, `fusion_history` (`list[dict]` per step with keys `FusionID`, `RadarID`, `InfraredID`, `DataLinkID`, `FusedState`, `FusedCov`, `FusionUpdateTime`), `ground_truth` (`list[list[dict]]`, each dict with `SphericalPositionNED` `[az_deg, el_deg, range_m]` and `CartesianPositionNED` `[x,y,z]` NED).

#### 2.2.3 Excel/CSV export output

Produced by `datalink.io.export.export_to_excel()` from a `summary_*.pkl` file. Output columns (one row per fused track per timestep, tracks with fewer than `min_track_points` rows dropped):

| Column | Type | Unit | Description |
|---|---|---|---|
| `flight_id` | int | — | Fusion ID |
| `targetID` | int/None | — | `FusedState`'s associated `TargetIndex`, if present in history |
| `time` | float | s | Fusion update time, rounded to 2 decimals |
| `latitude` | float | deg | Absolute geodetic latitude of the fused track |
| `longitude` | float | deg | Absolute geodetic longitude |
| `altitude` | float | m | Absolute altitude (ellipsoidal, via `ecef_to_lla`) |
| `track` | float | deg | Ground track/heading, from the fused velocity vector |
| `ground_speed` | float | m/s | Absolute ground speed (fused + ownship velocity) |
| `altitudeRate` | float | m/s | Absolute vertical rate |
| `headingRate` | float | deg/s | Wrapped heading rate |
| `type` | str | — | From `fusion_history[i]['Type']` if present, else `""` |

Output format is chosen by the `output_path` extension: `.xlsx` (via `pandas`/`openpyxl`, falls back to CSV content with an `.xlsx` name if `pandas` is unavailable) or `.csv`.

### 2.3 Trajectory Input Interface

Consumed by `Simulation` / `Ownship`. Two shapes are used depending on the code path:

1. **Loaded from file** (`.mat`/`.npy`/`.npz`/`.pkl` via `_load_trajectory`): a `list` of `types.SimpleNamespace`, one per platform, each with attributes `id`, `time`, `latitude`, `longitude`, `altitude`, `heading`, `pitch`, `roll` (all 1-D `float` arrays, same length as `time`).
2. **Per-ownship `trajectory` object** passed into `Ownship.__init__` (`self.platform`): expected to expose `.Position` (`(3,)` array) and `.Trajectory.Velocity` / `.Trajectory.Acceleration` (`(3,)` arrays), consumed in `Ownship._get_ir_detections`.

---

## 3. Internal Module Interfaces

### 3.1 `datalink.transformations`

```python
cartesian_to_msc(x: np.ndarray) -> np.ndarray
```
Input: Cartesian state `[x, vx, y, vy, z, vz]`, shape `(6,)` or `(6, N)`. Output: MSC state `[az, omega, el, elDot, invR, dRinvR]`, same shape. Angles in **radians**.

```python
msc_to_cartesian(msc: np.ndarray) -> np.ndarray
```
Inverse of the above. Same shape rules.

```python
cart2sph_gradient(states: np.ndarray) -> np.ndarray        # (4, 6)
cartesian_to_msc_gradient(cart: np.ndarray) -> np.ndarray  # (6, 6)
msc_to_cartesian_gradient(state: np.ndarray) -> np.ndarray # (6, 6)
const_vel_jac(dt: float) -> tuple[np.ndarray, np.ndarray]  # (6,6), (6,3)
```
All take/return `np.float64` arrays; `states`/`cart`/`state` inputs are 1-D length-6 vectors (`.ravel()`'d internally, so `(6,1)` also works).

### 3.2 `datalink.io.geo`

All lat/lon in **degrees**, all distances in **metres** — note this is the opposite convention from the MSC angles above (radians).

```python
lla_to_ecef(lat, lon, alt) -> tuple[x, y, z]                    # deg, deg, m → m, m, m
ecef_to_lla(x, y, z) -> tuple[lat_deg, lon_deg, alt_m]
ecef_to_ned(u, v, w, lat0, lon0) -> tuple[N, E, D]               # differential only
ned_to_ecef(x_east, y_north, z_up, lat0, lon0, h0) -> tuple[x, y, z]
geodetic_to_ned(lat, lon, alt, lat0, lon0, alt0) -> tuple[N, E, D]
ned_state_to_lla(ned_state, lat0, lon0, alt0) -> tuple[lat, lon, alt]
```
`ned_state_to_lla` expects a 6-element Cartesian CV state `[x, vx, y, vy, z, vz]` and reads `ned_state[2]` as East, `ned_state[0]` as North, `ned_state[4]` passed **directly** as `z_up` (i.e. NED-Down is **not** negated before being passed as `z_up` — see §4 hazard note).

### 3.3 `datalink.sensors.radar.detection.Detection` (dataclass)

| Field | Type | Default | Notes |
|---|---|---|---|
| `time` | `float` | — (required, no default) | Simulation time of the detection |
| `measurement` | `np.ndarray` | — (required) | Position `[x,y,z]` or `[az,el,r]` depending on frame |
| `sensor_index` | `int` | `0` | |
| `measurement_noise` | `np.ndarray` | `eye(3)` | Covariance |
| `measurement_parameters` | `dict` | `{}` | e.g. `{'Frame': 'spherical'}` |
| `object_attributes` | `dict` | `{}` | e.g. `SNR`, `TargetIndex`, `FoF` |

**Hazard:** `time` has no default and is a required positional/keyword field — constructing `Detection(measurement=..., measurement_noise=..., measurement_parameters=...)` without `time` raises `TypeError`. (`KULLANIM_KILAVUZU.txt`'s §2 example omits it — this is a documentation bug carried over from the guide, corrected in this ICD and in the README.)

### 3.4 `datalink.sensors.radar.track.Track` (dataclass)

| Field | Type | Default |
|---|---|---|
| `track_id` | `int` | — required |
| `state` | `np.ndarray` `(6,)` | — required |
| `state_covariance` | `np.ndarray` `(6,6)` | — required |
| `update_time` | `float` | `0.0` |
| `age` | `int` | `1` |
| `is_confirmed` | `bool` | `False` |
| `object_attributes` | `dict` | `{}` |
| `missed_count` | `int` | `0` |
| `hit_count` | `int` | `1` |

Properties: `.position -> state[[0,2,4]]`, `.velocity -> state[[1,3,5]]`.

### 3.5 `datalink.sensors.infrared`

```python
class MSCEKF:
    def __init__(self, state, covariance, process_noise, measurement_noise, observer_input=None)
    def predict(self, dt: float) -> None
    def update(self, measurement, measurement_noise=None) -> np.ndarray   # innovation, shape (2,)
```
`state`: MSC `(6,)`. `process_noise`: Cartesian acceleration `(3,3)`. `measurement_noise`: `(2,2)`, radians². `update(measurement)` expects **`measurement.shape == (2,)`**, i.e. `[az_rad, el_rad]` — **not** a 4-vector. (`KULLANIM_KILAVUZU.txt`'s §2 example calls `ekf.update(measurement)` with a 4-element array — this fails; only `[az, el]` is accepted.)

```python
init_cv_msc_ekf(detection: dict, range_estimation: tuple | None = None) -> MSCEKF
init_msc_rp_ekf(detection: dict, r_min=8e3, r_max=8e4, num_filters=3) -> list[MSCEKF]
```
**`detection` here is a plain `dict`, not a `datalink.sensors.radar.detection.Detection` object**, with keys `measurement` (`[az_deg, el_deg]`, **degrees**), `measurement_noise` (`(2,2)`, degrees²), `orientation` (`(3,3)`, default `eye(3)`), `sensor_velocity` (`(3,)`, default zeros).

### 3.6 `datalink.sensors.radar.tracking`

```python
class CVEKF:
    def __init__(self, state, state_covariance, process_noise=None, measurement_noise=None)
    def predict(self, dt: float) -> None
    def update(self, measurement, measurement_noise=None) -> np.ndarray   # innovation, shape (3,)

init_cv_ekf(detection: Detection) -> CVEKF   # takes a real Detection object, Cartesian [x,y,z]
```

```python
class GNNTracker:
    def __init__(self, max_tracks=100, gate=1e4,
                 confirm_threshold=(2,3), delete_threshold=(3,4),
                 has_cost_matrix_input=False)
    def step(self, detections: list[Detection], sim_time: float,
             cost_matrix: np.ndarray | None = None,
             detectable_ids: list[int] | None = None,
             dt: float = 0.2) -> tuple[list[Track], list[Track]]   # (confirmed, all)
    def delete_track(self, track_id: int) -> None
    def set_process_noise(self, track_id: int, Q: np.ndarray) -> None
    def is_locked(self) -> bool
```
**Hazard:** the constructor parameters are `confirm_threshold`/`delete_threshold` (tuples), and stepping is done via `.step(detections, sim_time, ...)`, **not** `.update(detections, dt=..., time=...)` and **not** `confirmation_threshold`/`deletion_threshold`. (`KULLANIM_KILAVUZU.txt`'s §3 example uses the wrong parameter names and the wrong method name — corrected in the README.)

```python
calculate_assignment_cost(detections: list[Detection], tracks: list[Track], gate_upper: float) -> np.ndarray  # (n_tracks, n_dets)
apply_dbscan(ned_detections, body_detections, epsilon: float, min_samples: int = 1) -> tuple[list[Detection], list[Detection]]
constvel(state, dt=1.0, noise=None) -> np.ndarray
set_observer_acceleration(acc: np.ndarray) -> None
get_observer_acceleration() -> np.ndarray
```

### 3.7 `datalink.fusion.fusion_manager`

```python
class FusionManager:
    def __init__(self, traj: Any, params: dict)   # params: full dict, 'scenarioParams' sub-dict is used
    def update_manager(self, confirmed_radar_tracks: list[Track],
                        confirmed_ir_tracks: list[Track], time: float) -> None
    def update_manager_with_datalink(self, datalink_tracks: list) -> None
    def get_fused_tracks(self) -> list[dict]
    # read-only properties: fusion_ids, radar_ids, ir_ids, datalink_ids,
    #                       fused_states, fused_covs, fusion_update_times
```

`get_fused_tracks()` return dict schema (one dict per entry with a valid fused state):

| Key | Type | Description |
|---|---|---|
| `fusion_id` | int | |
| `state` | `np.ndarray (6,)` | Cartesian `[x,vx,y,vy,z,vz]` |
| `state_covariance` | `np.ndarray (6,6)` | |
| `update_time` | float | |
| `radar_id` | int | `-1` if no radar contribution |
| `ir_id` | int | `-1` if no IR contribution |
| `dl_id` | int | `-1` if no DataLink contribution |
| `target_index` | int | |
| `is_fused` | bool | `True` only if both radar and IR contributed |
| `is_ir_only` | bool | present (`True`) only in the IR-only branch, see below |

When only an IR-only fused state exists (no full radar+IR fusion yet), the entry instead carries `state`/`state_covariance`/`update_time` from `fused_state_ir_only`/`fused_cov_ir_only`/`fusion_update_time_ir_only`, `is_fused=False`, `is_ir_only=True`.

`FusionEntry` (dataclass) — internal per-target record; relevant fields: `fusion_id, radar_id, ir_id, dl_id, radar_state, radar_cov, radar_update_time, ir_state, ir_cov, ir_update_time, dl_state, dl_cov, dl_update_time, fused_state, fused_cov, fusion_update_time, fused_state_ir_only, fused_cov_ir_only, fusion_update_time_ir_only, cross_cov, target_index` plus per-step history lists (`radar_updates`, `ir_updates`, `fusion_updates`, `dl_updates`, `virtual_updates`, `*_ids_history`, `*_state_history`, capped at 50 entries).

### 3.8 `datalink.core.ownship.Ownship`

```python
class Ownship:
    def __init__(self, platform_id: int, trajectory: Any, params: dict,
                 blue1_idx: int, initial_pos: np.ndarray, initial_lla: np.ndarray,
                 num_ownship: int, num_target: int)
    def produce_sensors_data(self, sim_time: float, traj: Any) -> None
    def step_radar(self, sim_time: float, delta_time: float) -> tuple[list[Detection], list[Track]]
    def step_infrared(self, sim_time: float, delta_time: float, ir_tracker: Any) -> tuple[list, list]
    def update_fusion(self, sim_time: float, confirmed_radar_tracks: list, confirmed_ir_tracks: list) -> list[dict]
    def update_fusion_datalink(self, datalink_tracks: list) -> None
```
Call order per step: `produce_sensors_data` → `step_radar` → `step_infrared` → `update_fusion`. `step_infrared` currently always returns `confirmed=[]` (IR tracker wiring is a stub — see §3.12).

### 3.9 `datalink.core.simulation.Simulation`

```python
class Simulation:
    def __init__(self, params_path: str, traj_path: str | None = None, enable_udp: bool = False)
    def run(self) -> list[dict]     # per-step history; see §2.2.2
    def close(self) -> None
```
Also usable as a context manager (`with Simulation(...) as sim:`) and from the CLI: `python -m datalink.core.simulation <params.json> [traj_path]`.

### 3.10 `datalink.error_metrics`

```python
ospa(X: np.ndarray, Y: np.ndarray, c=100.0, p=2.0, x_ids=None, y_ids=None) -> tuple[float,float,float,float]
    # (total, localisation, cardinality, labeling)
ospa_cartesian(tracks: list, truths: list, c=100.0, p=2.0) -> tuple[float,float,float,float]
ospa_msc(tracks: list, truths: list, c=0.1, p=2.0) -> tuple[float,float,float,float]
    # angular distance on [az, el] (radians), indices 0 and 2 of MSC state
hausdorff(P: np.ndarray, Q: np.ndarray) -> tuple[float, np.ndarray]   # (distance, pairwise matrix)
directed_hausdorff(P, Q) -> float
calculate_optimization_errors(hist: dict, ownship_data: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    # (fusion_timestep, fusion_pos_error[m], fusion_az_error[deg], fusion_el_error[deg]), each shape (T, N)
calculate_radar_ospa(confirmed_radar_tracks, true_tracks, c=100.0, p=2.0) -> tuple[float,float,float,float]
calculate_ir_ospa(confirmed_ir_tracks, true_tracks, c=0.1, p=2.0) -> tuple[float,float,float,float]
calculate_fusion_ospa(fused_tracks, true_tracks, c=100.0, p=2.0) -> tuple[float,float,float,float]
```

### 3.11 `datalink.io.export.export_to_excel`

```python
export_to_excel(summary_path: str, ownship_data: Any,
                 output_path: str = "SenaryoCSV/output.xlsx",
                 min_track_points: int = 10) -> None
```
`ownship_data` must expose 1-D arrays `.time, .latitude, .longitude, .altitude, .heading, .xSpeed, .ySpeed, .zSpeed`. See §2.2.3 for output columns. Also runnable via CLI: `python -m datalink.io.export <summary.pkl> <output.xlsx>`.

### 3.12 `datalink.sensors.infrared.infrared_sensor.InfraredSensor`

```python
class InfraredSensor:
    def __init__(self, sensor_index: int, params: dict)   # params = {'infraredSensor': {...}}
    def step(self, targets: list, ins: Any, time: float) -> tuple[list[IRDetection], Any, Any]
    def is_locked(self) -> bool
    def release(self) -> None
```
`ins` (own-platform state passed to `.step`) is expected to expose `.Position (3,)`, `.Velocity (3,)`, `.Orientation (3,3 DCM)`, matching the `SimpleNamespace` built in `Ownship._get_ir_detections`. Returns angle-only `IRDetection` objects (azimuth/elevation in degrees). Currently used as a stub — `Ownship.step_infrared` collects raw detections but does not yet run a tracker over them (`confirmed_ir_tracks` is always `[]` — full IR tracker wiring, GNN+MSC-EKF, is noted in the source as future work).

### 3.13 Radar subsystem — brief interfaces

| Module | Class/function | Role |
|---|---|---|
| `sensors.radar.radar_sensor` | `RadarSensor(blue1_index, sensor_index, params)` | Top-level radar wrapper; `.get_detections_and_tracks(...) -> (body_dets, tracker_dets, tracks, coverage_config)`; sweeps the FOV in `n_clusters=5` sub-beams per step |
| `sensors.radar.mpar` | `MPAR(sensor_index, params)` | Wraps `TrackingRadar` + `GNNTracker`; `.update_sensor(...)`; owns DBSCAN pre-filtering and orthogonal-ghost-track removal |
| `sensors.radar.tracking_radar` | `TrackingRadar(params)` | Generates noisy pseudo-measurements per target from `trackingRadar`+`clutterParams`; `.calculate_pseudo_measurements(...)` |
| `sensors.radar.clutter` | `Clutter(params)` | Ground-clutter scatterer sampling and SCR (signal-to-clutter ratio) calculation |
| `sensors.radar.snjr` | `radar_snjr_db(...)` | Radar equation → signal-to-noise-plus-jamming ratio in dB |
| `sensors.radar.target` | `Target(target_id, pose_in_ownship, pose_in_local, rcs_avg, swerling_type, chaff)` | Per-target pose/RCS bookkeeping, spherical/Cartesian conversions |
| `sensors.radar.radar_beam` | `RadarBeam(prf, tau, theta, phi, theta_res, phi_res)` | Tracks current beam pointing angle |
| `sensors.radar.fluctuation_models` | `FluctuationModel(swerling_type, sigma_avg)` | Swerling 0–5 RCS sampling (`.sample()`) |
| `sensors.radar.jammers.jammer` | `Jammer(params)` | EW jammer model attached to a target |
| `sensors.radar.simulation_history` | `SimulationHistory()` | Per-step snapshot recorder for replay/export (`.append(...)`, iterator-style `.next()`) |

---

## 4. Coordinate System & Units Conventions

### 4.1 Cartesian CV state vector

`[x, vx, y, vy, z, vz]` — NED metres, position at indices `[0,2,4]`, velocity at `[1,3,5]`.

### 4.2 MSC state vector

`[az, omega, el, elDot, invR, dRinvR]` — **radians** and **1/metres**:
- `az` — azimuth [rad]
- `omega` — azimuth rate, horizontal-plane-projected [rad/s]
- `el` — elevation [rad]
- `elDot` — elevation rate [rad/s]
- `invR` — inverse range [1/m]
- `dRinvR` — range-rate / range [1/s]

### 4.3 Interface hazard — mixed angle units

**`datalink.io.geo`** (LLA/ECEF/NED conversions) uses **degrees** throughout. **`datalink.transformations`** (Cartesian↔MSC) uses **radians** throughout. **UDP telemetry** (§2.1) transmits azimuth/elevation in **degrees** (converted from the radian MSC state at the point of message formatting). **`SimulationSummary`**'s stored `ir_tracks` are also pre-converted to degrees, while `radar_tracks`/`fusion_history['FusedState']` remain in NED metres/radians as native Cartesian/MSC state. A caller crossing between the `geo` module and the `transformations`/tracking modules must convert explicitly — no implicit unit conversion happens at those boundaries.

**`datalink.io.geo.ned_state_to_lla`** (and `export._ned_state_to_lla`) pass `ned_state[4]` (Z, i.e. NED-Down) **directly** as `z_up` without sign-flipping, per the fixed calling convention documented in the source comments (`io/geo.py:221-235`). Implementers integrating with this function should treat this as the defined (if non-obvious) behavior rather than a bug to "fix".

### 4.4 Swerling fluctuation model naming

| Name(s) | Distribution |
|---|---|
| `'Swerling0'`, `'Swerling5'` | Constant RCS |
| `'Swerling1'`, `'Swerling2'` | `Gamma(shape=1, scale=σ)` |
| `'Swerling3'`, `'Swerling4'` | `Gamma(shape=2, scale=σ/2)` |

---

## 5. Revision History

| Rev | Date | Description |
|---|---|---|
| 1.0 | 2026-09-27 | Initial draft, generated from source-code inspection of the `datalink` package (all field/method tables verified against `datalink/*.py` as of this date). |
