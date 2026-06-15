from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

EARTH_RADIUS_M = 6_371_000.0
STATE_NAME = {0: "line", 1: "curve", 2: "line", 3: "line", 4: "circle"}
STATE_STYLE_NAME = {0: "Line", 1: "Curve", 2: "Line", 3: "Line", 4: "Circle"}
CLASS_COLORS = {
    "line": "#1f77b4",
    "curve": "#ff7f0e",
    "circle": "#d62728",
}


@dataclass(frozen=True)
class AnalysisResult:
    output_dir: Path
    split_dir: Path
    overview_png: Path
    full_result_csv: Path
    segment_csvs: list[Path]


def analyze_csv(csv_path: Path | str, output_dir: Path | str = "output", sliding_window: int = 30) -> AnalysisResult:
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    split_dir = output_dir / "splited"
    output_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    original, dataframe = _load_drone_csv(csv_path)
    dataframe = _engineer_features(dataframe, max(1, int(sliding_window or 30)))
    dataframe = _classify_states(dataframe)
    dataframe = _post_process_states(dataframe)

    full_result_csv = output_dir / "full_result.csv"
    export_dataframe = _build_export_dataframe(original, dataframe)
    export_dataframe.to_csv(full_result_csv, index=False, float_format="%.12f")

    segment_csvs = _write_segment_csvs(original, dataframe, split_dir)
    overview_png = output_dir / "AI Analyze Overview.png"
    _save_overview_png(dataframe, overview_png)

    return AnalysisResult(
        output_dir=output_dir.resolve(),
        split_dir=split_dir.resolve(),
        overview_png=overview_png.resolve(),
        full_result_csv=full_result_csv.resolve(),
        segment_csvs=[path.resolve() for path in segment_csvs],
    )


def _load_drone_csv(csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    original = pd.read_csv(csv_path, header=None, dtype=str)
    numeric = original.apply(pd.to_numeric, errors="coerce")
    valid_start_idx = -1
    for column_index in range(min(numeric.shape[1], 5)):
        value = numeric.iloc[0, column_index]
        if pd.notna(value) and float(value) > 500_000:
            valid_start_idx = column_index
            break

    if valid_start_idx != -1 and numeric.shape[1] >= valid_start_idx + 4:
        dataframe = numeric.iloc[:, [valid_start_idx, valid_start_idx + 1, valid_start_idx + 2, valid_start_idx + 3]].copy()
        dataframe.columns = ["Time", "Longitude", "Latitude", "Altitude"]
    elif valid_start_idx != -1 and numeric.shape[1] >= valid_start_idx + 3:
        dataframe = numeric.iloc[:, [valid_start_idx, valid_start_idx + 1, valid_start_idx + 2]].copy()
        dataframe.columns = ["Time", "Longitude", "Latitude"]
        dataframe["Altitude"] = 0.0
    elif numeric.shape[1] >= 3:
        dataframe = numeric.iloc[:, [0, 1, 2]].copy()
        dataframe.columns = ["Time", "Longitude", "Latitude"]
        dataframe["Altitude"] = 0.0
    else:
        raise ValueError("CSV must contain at least Time, Longitude, and Latitude columns.")

    dataframe["Latitude"] = pd.to_numeric(dataframe["Latitude"], errors="coerce")
    dataframe["Longitude"] = pd.to_numeric(dataframe["Longitude"], errors="coerce")
    dataframe["Altitude"] = pd.to_numeric(dataframe["Altitude"], errors="coerce").fillna(0.0)
    dataframe = dataframe.dropna(subset=["Latitude", "Longitude"])
    original = original.loc[dataframe.index].copy()
    if dataframe.empty:
        raise ValueError("No valid Latitude/Longitude rows were found in the CSV.")
    return original, dataframe


def _engineer_features(dataframe: pd.DataFrame, window_size: int) -> pd.DataFrame:
    dataframe = dataframe.copy()
    smooth_count = 5
    dataframe["Latitude_s"] = dataframe["Latitude"].rolling(smooth_count, center=True).mean().bfill().ffill()
    dataframe["Longitude_s"] = dataframe["Longitude"].rolling(smooth_count, center=True).mean().bfill().ffill()

    lat_rad = np.deg2rad(dataframe["Latitude_s"])
    lon_rad = np.deg2rad(dataframe["Longitude_s"])
    delta_x = EARTH_RADIUS_M * lon_rad.diff().fillna(0) * np.cos(lat_rad.mean())
    delta_y = EARTH_RADIUS_M * lat_rad.diff().fillna(0)
    delta_z = dataframe["Altitude"].diff().fillna(0)
    distance = np.sqrt(delta_x**2 + delta_y**2 + delta_z**2)

    heading_rad = np.where(distance > 0.1, np.arctan2(delta_y, delta_x), np.nan)
    dataframe["Pseudo_Yaw"] = pd.Series(np.rad2deg(heading_rad), index=dataframe.index).ffill().bfill().fillna(0)
    dataframe["Yaw_unwrap"] = np.rad2deg(np.unwrap(np.deg2rad(dataframe["Pseudo_Yaw"])))
    dataframe["Yaw_diff"] = dataframe["Yaw_unwrap"].diff().fillna(0).abs()
    dataframe["Yaw_rate"] = dataframe["Yaw_diff"].rolling(window=window_size, center=True).mean().fillna(0)
    dataframe["Yaw_range"] = (
        dataframe["Yaw_unwrap"].rolling(window=window_size, center=True).max()
        - dataframe["Yaw_unwrap"].rolling(window=window_size, center=True).min()
    ).fillna(0)
    dataframe["Speed"] = distance
    dataframe["Speed_std"] = dataframe["Speed"].rolling(window=window_size, center=True).std().fillna(0)
    dataframe["dx"] = delta_x
    dataframe["dy"] = delta_y
    dataframe["ds"] = distance
    dataframe["Local_Tort"] = _local_tortuosity_multi(dataframe, [80, 120, 160])
    dataframe["AbsTurn"] = _windowed_abs_turn(dataframe["Yaw_unwrap"].values, 200)

    delta_dx = dataframe["dx"].diff().fillna(0)
    delta_dy = dataframe["dy"].diff().fillna(0)
    distance_xy = np.sqrt(dataframe["dx"]**2 + dataframe["dy"]**2)
    curvature = np.abs(dataframe["dx"] * delta_dy - dataframe["dy"] * delta_dx) / (distance_xy**3 + 1e-6)
    dataframe["Curvature"] = curvature.rolling(window=30, center=True).mean().fillna(0)

    yaw_diff_signed = dataframe["Yaw_unwrap"].diff().fillna(0)
    zero_crossings = ((yaw_diff_signed > 0.05).astype(int).diff().fillna(0) != 0).astype(int)
    dataframe["ZCR"] = zero_crossings.rolling(window=100, center=True).sum().fillna(0)
    yaw_mean = yaw_diff_signed.rolling(window=100, center=True).mean().fillna(0)
    yaw_std = yaw_diff_signed.rolling(window=100, center=True).std().fillna(0)
    dataframe["Periodicity"] = dataframe["ZCR"] * (yaw_std / (yaw_mean.abs() + 0.1))
    return dataframe


def _local_tortuosity_multi(dataframe: pd.DataFrame, windows: list[int]) -> np.ndarray:
    row_count = len(dataframe)
    delta_x = dataframe["dx"].values
    delta_y = dataframe["dy"].values
    distance = dataframe["ds"].values
    cumulative_x = np.concatenate(([0], np.cumsum(delta_x)))
    cumulative_y = np.concatenate(([0], np.cumsum(delta_y)))
    cumulative_distance = np.concatenate(([0], np.cumsum(distance)))
    tortuosity = np.ones(row_count)
    for window in windows:
        half_window = window // 2
        for row_index in range(row_count):
            start_index = max(0, row_index - half_window)
            end_index = min(row_count - 1, row_index + half_window)
            path_length = cumulative_distance[end_index + 1] - cumulative_distance[start_index]
            net_x = cumulative_x[end_index + 1] - cumulative_x[start_index]
            net_y = cumulative_y[end_index + 1] - cumulative_y[start_index]
            net_distance = np.sqrt(net_x**2 + net_y**2)
            if net_distance > 1.0:
                value = path_length / net_distance
            elif path_length > 3.0:
                value = 5.0
            else:
                value = 1.0
            if value > tortuosity[row_index]:
                tortuosity[row_index] = value
    return tortuosity


def _windowed_abs_turn(yaw_unwrap: np.ndarray, window: int) -> np.ndarray:
    row_count = len(yaw_unwrap)
    yaw_delta = np.concatenate(([0], np.abs(np.diff(yaw_unwrap))))
    cumulative_yaw_delta = np.concatenate(([0], np.cumsum(yaw_delta)))
    abs_turn = np.zeros(row_count)
    half_window = window // 2
    for row_index in range(row_count):
        start_index = max(0, row_index - half_window)
        end_index = min(row_count - 1, row_index + half_window)
        abs_turn[row_index] = cumulative_yaw_delta[end_index + 1] - cumulative_yaw_delta[start_index]
    return abs_turn


def _classify_states(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = dataframe.copy()
    feature_cols = ["Yaw_rate", "Yaw_range", "Local_Tort", "AbsTurn", "Curvature", "Periodicity"]
    dataframe[feature_cols] = dataframe[feature_cols].fillna(0)
    if len(dataframe) < 5:
        dataframe["State"] = 0
        dataframe["State_Smooth"] = 0
        return dataframe

    scaled_features = StandardScaler().fit_transform(dataframe[feature_cols])
    raw_state_1 = _predict_clusters(scaled_features, component_count=2)

    state_means = {
        state: np.mean(dataframe.loc[raw_state_1 == state, "Local_Tort"]) if np.any(raw_state_1 == state) else 0.0
        for state in range(2)
    }
    straight_label = min(state_means.keys(), key=lambda state: state_means[state])
    others_label = 1 - straight_label
    state_array = np.zeros(len(dataframe), dtype=int)
    others_mask = raw_state_1 == others_label
    others_indices = np.where(others_mask)[0]

    if len(others_indices) > 0:
        if len(others_indices) < 5:
            state_array[others_indices] = 1
        else:
            scaled_others = StandardScaler().fit_transform(dataframe.loc[others_mask, feature_cols])
            split_indices = np.where(np.diff(others_indices) > 1)[0] + 1
            blocks = np.split(others_indices, split_indices)
            lengths = [len(block) for block in blocks]
            raw_state_2 = _predict_clusters(scaled_others, component_count=2, lengths=lengths)
            turn_means = {}
            for state in range(2):
                original_indices = others_indices[raw_state_2 == state]
                turn_means[state] = np.mean(dataframe.loc[original_indices, "AbsTurn"]) if len(original_indices) else 0.0
            rotate_label = max(turn_means.keys(), key=lambda state: turn_means[state])
            curve_label = 1 - rotate_label
            state_array[others_indices[raw_state_2 == curve_label]] = 1
            state_array[others_indices[raw_state_2 == rotate_label]] = 2

    dataframe["State"] = state_array
    tort_hotspot = dataframe["Local_Tort"].values >= 2.0
    state_smooth = np.zeros(len(dataframe), dtype=int)
    state_smooth[state_array == 2] = 2
    state_smooth[(state_array == 1) | tort_hotspot] = 1
    dataframe["State_Smooth"] = state_smooth
    return dataframe


def _predict_clusters(features: np.ndarray, component_count: int, lengths: list[int] | None = None) -> np.ndarray:
    try:
        from hmmlearn import hmm

        model = hmm.GaussianHMM(n_components=component_count, covariance_type="diag", n_iter=100, random_state=42)
        if lengths is None:
            model.fit(features)
        else:
            model.fit(features, lengths=lengths)
        return model.predict(features)
    except ImportError:
        model = KMeans(n_clusters=component_count, n_init=10, random_state=42)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="KMeans is known to have a memory leak on Windows with MKL.*")
            return model.fit_predict(features)


def _post_process_states(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = dataframe.copy()
    dataframe["State_Final"] = dataframe["State_Smooth"].copy()
    dataframe.loc[dataframe["State_Smooth"] == 2, "State_Final"] = 4
    dataframe["PCA_Ratio"] = 0.0
    _verify_curves_with_pca(dataframe, linearity_threshold=0.98)
    _refine_line_boundaries(dataframe, dist_threshold=1.0, angle_threshold=10.0)
    _resegment(dataframe)
    _label_figure8(dataframe, turn_threshold=400.0, tort_threshold=2.5, gap=450)
    _reindex_segment_ids(dataframe)
    return dataframe


def _is_linear_path(lats: np.ndarray, lons: np.ndarray, linearity_threshold: float = 0.95) -> tuple[bool, float]:
    if len(lats) < 6:
        return False, 0.0
    lat_rad = np.deg2rad(np.mean(lats))
    xs = np.deg2rad(lons) * EARTH_RADIUS_M * np.cos(lat_rad)
    ys = np.deg2rad(lats) * EARTH_RADIUS_M
    coords = np.column_stack([xs, ys])
    span = np.ptp(coords, axis=0)
    if np.max(span) < 1.0:
        return True, 1.0
    pca = PCA(n_components=2)
    pca.fit(coords)
    ratio_pc1 = float(pca.explained_variance_ratio_[0])
    return ratio_pc1 >= linearity_threshold, ratio_pc1


def _verify_curves_with_pca(dataframe: pd.DataFrame, linearity_threshold: float = 0.95) -> None:
    states = dataframe["State_Final"].values.copy()
    row_count = len(states)
    current_index = 0
    while current_index < row_count:
        state = states[current_index]
        run_end = current_index
        while run_end < row_count and states[run_end] == state:
            run_end += 1
        if state in (1, 4):
            lats = dataframe["Latitude"].values[current_index:run_end]
            lons = dataframe["Longitude"].values[current_index:run_end]
            is_linear, ratio = _is_linear_path(lats, lons, linearity_threshold)
            dataframe.iloc[current_index:run_end, dataframe.columns.get_loc("PCA_Ratio")] = ratio
            if is_linear:
                states[current_index:run_end] = 0
        current_index = run_end
    dataframe["State_Final"] = states


def _refine_line_boundaries(dataframe: pd.DataFrame, dist_threshold: float = 1.5, angle_threshold: float = 15.0) -> None:
    states = dataframe["State_Final"].values.copy()
    row_count = len(states)
    if row_count < 50:
        return

    lat_rad = np.deg2rad(dataframe["Latitude_s"])
    lon_rad = np.deg2rad(dataframe["Longitude_s"])
    delta_x = EARTH_RADIUS_M * lon_rad.diff().fillna(0) * np.cos(lat_rad.mean())
    delta_y = EARTH_RADIUS_M * lat_rad.diff().fillna(0)
    x_values = delta_x.cumsum().values
    y_values = delta_y.cumsum().values
    gradient_x = np.gradient(x_values)
    gradient_y = np.gradient(y_values)
    headings = np.column_stack([gradient_x, gradient_y])
    norms = np.linalg.norm(headings, axis=1, keepdims=True)
    headings = np.divide(headings, norms, out=np.zeros_like(headings), where=norms > 1e-5)

    run_starts, run_states, run_lengths = _state_runs(states)
    fit_count = 30
    for run_index, state in enumerate(run_states):
        if state != 0:
            continue
        run_start = run_starts[run_index]
        run_length = run_lengths[run_index]
        run_end = run_start + run_length
        if run_length < 15:
            continue

        if run_index > 0 and run_states[run_index - 1] in (1, 4):
            previous_start = run_starts[run_index - 1]
            fit_size = min(fit_count, run_length)
            mean_coords, line_dir = _fit_line_direction(x_values[run_start:run_start + fit_size], y_values[run_start:run_start + fit_size])
            for point_index in range(run_start - 1, previous_start - 1, -1):
                if _point_matches_line(point_index, x_values, y_values, headings, mean_coords, line_dir, dist_threshold, angle_threshold):
                    states[point_index] = 0
                else:
                    break

        if run_index < len(run_starts) - 1 and run_states[run_index + 1] in (1, 4):
            next_start = run_starts[run_index + 1]
            next_end = next_start + run_lengths[run_index + 1]
            fit_size = min(fit_count, run_length)
            mean_coords, line_dir = _fit_line_direction(x_values[run_end - fit_size:run_end], y_values[run_end - fit_size:run_end])
            for point_index in range(run_end, next_end):
                if _point_matches_line(point_index, x_values, y_values, headings, mean_coords, line_dir, dist_threshold, angle_threshold):
                    states[point_index] = 0
                else:
                    break
    dataframe["State_Final"] = states


def _state_runs(states: np.ndarray) -> tuple[list[int], list[int], list[int]]:
    run_starts: list[int] = []
    run_states: list[int] = []
    run_lengths: list[int] = []
    current_state = states[0]
    current_start = 0
    for row_index in range(1, len(states)):
        if states[row_index] != current_state:
            run_starts.append(current_start)
            run_states.append(current_state)
            run_lengths.append(row_index - current_start)
            current_state = states[row_index]
            current_start = row_index
    run_starts.append(current_start)
    run_states.append(current_state)
    run_lengths.append(len(states) - current_start)
    return run_starts, run_states, run_lengths


def _fit_line_direction(x_values: np.ndarray, y_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords = np.column_stack([x_values, y_values])
    mean_coords = np.mean(coords, axis=0)
    centered = coords - mean_coords
    _, _, vectors = np.linalg.svd(centered, full_matrices=False)
    return mean_coords, vectors[0]


def _point_matches_line(
    point_index: int,
    x_values: np.ndarray,
    y_values: np.ndarray,
    headings: np.ndarray,
    mean_coords: np.ndarray,
    line_dir: np.ndarray,
    dist_threshold: float,
    angle_threshold: float,
) -> bool:
    point = np.array([x_values[point_index], y_values[point_index]])
    diff = point - mean_coords
    proj = np.dot(diff, line_dir) * line_dir
    perp = diff - proj
    dist = np.linalg.norm(perp)
    heading = headings[point_index]
    cos_sim = abs(np.dot(heading, line_dir))
    angle_diff = np.arccos(min(1.0, cos_sim)) * 180.0 / np.pi
    return bool(dist < dist_threshold and angle_diff < angle_threshold)


def _resegment(dataframe: pd.DataFrame) -> None:
    states = dataframe["State_Final"].values
    changes = np.concatenate(([1], (states[1:] != states[:-1]).astype(int)))
    dataframe["Final_Segment_ID"] = np.cumsum(changes).astype(int)


def _label_figure8(dataframe: pd.DataFrame, turn_threshold: float, tort_threshold: float, gap: int) -> None:
    strong = (dataframe["AbsTurn"] >= turn_threshold) & (dataframe["Local_Tort"] >= tort_threshold)
    strong_indices = np.where(strong)[0]
    if len(strong_indices) == 0:
        return
    groups: list[tuple[int, int]] = []
    group_start = strong_indices[0]
    group_end = strong_indices[0]
    for index_value in strong_indices[1:]:
        if index_value - group_end <= gap:
            group_end = index_value
        else:
            groups.append((group_start, group_end))
            group_start = index_value
            group_end = index_value
    groups.append((group_start, group_end))

    row_count = len(dataframe)
    for start_index, end_index in groups:
        while start_index > 0 and dataframe.iloc[start_index - 1]["State_Final"] in (1, 4):
            start_index -= 1
        while end_index < row_count - 1 and dataframe.iloc[end_index + 1]["State_Final"] in (1, 4):
            end_index += 1
        dataframe.iloc[start_index:end_index + 1, dataframe.columns.get_loc("State_Final")] = 4
    _resegment(dataframe)


def _reindex_segment_ids(dataframe: pd.DataFrame) -> None:
    alive_ids = sorted(dataframe[dataframe["Final_Segment_ID"] != -1]["Final_Segment_ID"].unique())
    id_mapping = {old_id: new_index for new_index, old_id in enumerate(alive_ids, start=1)}
    dataframe.loc[dataframe["Final_Segment_ID"] != -1, "Final_Segment_ID"] = (
        dataframe.loc[dataframe["Final_Segment_ID"] != -1, "Final_Segment_ID"].map(id_mapping).astype(int)
    )


def _build_export_dataframe(original: pd.DataFrame, dataframe: pd.DataFrame) -> pd.DataFrame:
    export = original.copy()
    for column_name in ["Yaw_rate", "Yaw_range", "Speed_std", "State", "Local_Tort", "AbsTurn", "PCA_Ratio", "Final_Segment_ID"]:
        export[column_name] = dataframe[column_name].values
    export["State_Final"] = [STATE_NAME.get(int(state), "unknown") for state in dataframe["State_Final"].values]
    return export


def _write_segment_csvs(original: pd.DataFrame, dataframe: pd.DataFrame, split_dir: Path) -> list[Path]:
    segment_paths: list[Path] = []
    for existing in split_dir.glob("*.csv"):
        existing.unlink()

    sorted_segment_ids = sorted(dataframe[dataframe["Final_Segment_ID"] != -1]["Final_Segment_ID"].unique())
    for segment_id in sorted_segment_ids:
        group = dataframe[dataframe["Final_Segment_ID"] == segment_id]
        if group.empty:
            continue
        state_final = int(group["State_Final"].iloc[0])
        label = STATE_NAME.get(state_final, f"state{state_final}")
        color = _segment_color(state_final)
        segment_export = _build_export_dataframe(original.loc[group.index].copy(), group)
        segment_export["Color"] = color
        segment_path = split_dir / f"{label}-{segment_id}.csv"
        segment_export.to_csv(segment_path, index=False, float_format="%.12f")
        segment_paths.append(segment_path)
    return segment_paths


def _segment_color(state_final: int) -> str:
    return CLASS_COLORS.get(STATE_NAME.get(state_final, "line"), "#000000")


def _save_overview_png(dataframe: pd.DataFrame, output_path: Path) -> None:
    sorted_segment_ids = sorted(dataframe[dataframe["Final_Segment_ID"] != -1]["Final_Segment_ID"].unique())
    fig, axis = plt.subplots(figsize=(11, 7))
    axis.plot(dataframe["Longitude"], dataframe["Latitude"], color="#cccccc", linewidth=1, alpha=0.5, label="Original Path")

    legend_labels: dict[str, str] = {}
    for segment_id in sorted_segment_ids:
        group = dataframe[dataframe["Final_Segment_ID"] == segment_id]
        state_final = int(group["State_Final"].iloc[0])
        color = _segment_color(state_final)
        line_width, line_style = _line_style(state_final)
        style_name = STATE_STYLE_NAME.get(state_final, f"State{state_final}")
        legend_labels[style_name] = color

        indices = group.index.values
        split_locations = np.where(np.diff(indices) > 1)[0] + 1
        sub_groups = np.split(indices, split_locations)
        for sub_indices in sub_groups:
            if len(sub_indices) < 2:
                continue
            sub_group = dataframe.loc[sub_indices]
            axis.plot(
                sub_group["Longitude"],
                sub_group["Latitude"],
                color=color,
                linewidth=line_width,
                linestyle=line_style,
            )

    axis.set_title("AI Analyze Overview", fontsize=15, fontweight="bold")
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    axis.grid(True, alpha=0.2)
    axis.set_aspect("equal", adjustable="datalim")
    _add_color_mapping_legend(fig, legend_labels)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(output_path, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def _line_style(state_final: int) -> tuple[int, str]:
    if state_final == 1:
        return 3, "--"
    if state_final == 4:
        return 4, "-"
    return 2, "-"


def _add_color_mapping_legend(fig: plt.Figure, legend_labels: dict[str, str]) -> None:
    if not legend_labels:
        return
    handles = [Patch(facecolor=color, edgecolor="#333333", label=label) for label, color in legend_labels.items()]
    fig.legend(
        handles=handles,
        title="Color Mapping for AI",
        loc="lower left",
        bbox_to_anchor=(0.01, 0.01),
        ncol=len(handles),
        frameon=True,
        fancybox=True,
        edgecolor="#dddddd",
        facecolor="#f8f8f8",
        fontsize=8,
        title_fontsize=8,
    )
