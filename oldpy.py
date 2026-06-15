import os
import json
import base64
import tempfile
import io
import httpx
import pandas as pd
import numpy as np
import folium
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm
from sklearn.decomposition import PCA as _PCA

from .utils import OPENAPI_KEY

def is_linear_path(lats, lons, linearity_threshold=0.95):
    if len(lats) < 6:
        return False, 0.0
        
    R = 6371000.0
    lat_rad = np.deg2rad(np.mean(lats))
    xs = np.deg2rad(lons) * R * np.cos(lat_rad)
    ys = np.deg2rad(lats) * R
    coords = np.column_stack([xs, ys])
    
    span = np.ptp(coords, axis=0)
    if np.max(span) < 1.0:  # Less than 1m movement 2D -> Stationary
        return True, 1.0
        
    pca = _PCA(n_components=2)
    pca.fit(coords)
    ratio_pc1 = pca.explained_variance_ratio_[0]
    return ratio_pc1 >= linearity_threshold, ratio_pc1


def verify_curves_with_pca(df_in, linearity_threshold=0.95):
    states = df_in['State_Final'].values.copy()
    n = len(states)
    if n == 0:
        return
        
    curr_idx = 0
    while curr_idx < n:
        state = states[curr_idx]
        run_end = curr_idx
        while run_end < n and states[run_end] == state:
            run_end += 1
        run_len = run_end - curr_idx
        
        if state in (1, 4):  # Curve (1) or Rotate (4)
            lats = df_in['Latitude'].values[curr_idx:run_end]
            lons = df_in['Longitude'].values[curr_idx:run_end]
            
            is_linear, ratio = is_linear_path(lats, lons, linearity_threshold)
            if is_linear:
                states[curr_idx:run_end] = 0
                
        curr_idx = run_end
        
    df_in['State_Final'] = states


def absorb_short_segments(df_in, min_len_dict):
    """
    df_in: DataFrame containing 'State_Final'
    min_len_dict: dict of {state: min_length}
    """
    states = df_in['State_Final'].values.copy()
    n = len(states)
    if n == 0:
        return
    
    curr_idx = 0
    while curr_idx < n:
        state = states[curr_idx]
        run_end = curr_idx
        while run_end < n and states[run_end] == state:
            run_end += 1
        run_len = run_end - curr_idx
        
        limit = min_len_dict.get(state, 20)
        if run_len < limit:
            left_state = states[curr_idx - 1] if curr_idx > 0 else None
            
            right_idx = run_end
            right_state = None
            right_len = 0
            if right_idx < n:
                right_state = states[right_idx]
                temp_idx = right_idx
                while temp_idx < n and states[temp_idx] == right_state:
                    temp_idx += 1
                right_len = temp_idx - right_idx
                
            left_len = 0
            if curr_idx > 0:
                temp_idx = curr_idx - 1
                while temp_idx >= 0 and states[temp_idx] == left_state:
                    temp_idx -= 1
                left_len = curr_idx - 1 - temp_idx
            
            if left_state is not None and right_state is not None:
                if left_state == right_state:
                    target_state = left_state
                else:
                    if left_len >= right_len:
                        target_state = left_state
                    else:
                        target_state = right_state
            elif left_state is not None:
                target_state = left_state
            elif right_state is not None:
                target_state = right_state
            else:
                curr_idx = run_end
                continue
                
            states[curr_idx:run_end] = target_state
            if curr_idx > 0:
                curr_idx = max(0, curr_idx - left_len)
            else:
                curr_idx = run_end
        else:
            curr_idx = run_end
            
    df_in['State_Final'] = states


def refine_line_boundaries(df_in, dist_threshold=1.5, angle_threshold=15.0):
    states = df_in['State_Final'].values.copy()
    n = len(states)
    if n < 50:
        return
        
    if 'X' not in df_in.columns or 'Y' not in df_in.columns:
        R = 6371000.0
        lat_rad = np.deg2rad(df_in['Latitude_s'])
        lon_rad = np.deg2rad(df_in['Longitude_s'])
        dx = R * lon_rad.diff().fillna(0) * np.cos(lat_rad.mean())
        dy = R * lat_rad.diff().fillna(0)
        X = dx.cumsum().values
        Y = dy.cumsum().values
    else:
        X = df_in['X'].values
        Y = df_in['Y'].values
        
    dx = np.gradient(X)
    dy = np.gradient(Y)
    headings = np.column_stack([dx, dy])
    norms = np.linalg.norm(headings, axis=1, keepdims=True)
    headings = np.where(norms > 1e-5, headings / norms, 0.0)
    
    run_starts = []
    run_states = []
    run_lengths = []
    curr_state = states[0]
    curr_start = 0
    for i in range(1, n):
        if states[i] != curr_state:
            run_starts.append(curr_start)
            run_states.append(curr_state)
            run_lengths.append(i - curr_start)
            curr_state = states[i]
            curr_start = i
    run_starts.append(curr_start)
    run_states.append(curr_state)
    run_lengths.append(n - curr_start)
    
    num_runs = len(run_starts)
    K_fit = 30
    
    for idx in range(num_runs):
        curr_state = run_states[idx]
        if curr_state != 0:
            continue
            
        r_start = run_starts[idx]
        r_len = run_lengths[idx]
        r_end = r_start + r_len
        
        if r_len < 15:
            continue
            
        # Left boundary (Curve-to-Line)
        if idx > 0 and run_states[idx-1] in (1, 4):
            prev_start = run_starts[idx-1]
            prev_len = run_lengths[idx-1]
            prev_end = prev_start + prev_len
            
            fit_size = min(K_fit, r_len)
            fit_X = X[r_start : r_start + fit_size]
            fit_Y = Y[r_start : r_start + fit_size]
            
            coords = np.column_stack([fit_X, fit_Y])
            mean_coords = np.mean(coords, axis=0)
            centered = coords - mean_coords
            u, s, vh = np.linalg.svd(centered, full_matrices=False)
            line_dir = vh[0]
            
            for p_idx in range(r_start - 1, prev_start - 1, -1):
                pt = np.array([X[p_idx], Y[p_idx]])
                diff = pt - mean_coords
                proj = np.dot(diff, line_dir) * line_dir
                perp = diff - proj
                dist = np.linalg.norm(perp)
                
                pt_heading = headings[p_idx]
                cos_sim = abs(np.dot(pt_heading, line_dir))
                angle_diff = np.arccos(min(1.0, cos_sim)) * 180.0 / np.pi
                
                if dist < dist_threshold and angle_diff < angle_threshold:
                    states[p_idx] = 0
                else:
                    break
                    
        # Right boundary (Line-to-Curve)
        if idx < num_runs - 1 and run_states[idx+1] in (1, 4):
            next_start = run_starts[idx+1]
            next_len = run_lengths[idx+1]
            next_end = next_start + next_len
            
            fit_size = min(K_fit, r_len)
            fit_X = X[r_end - fit_size : r_end]
            fit_Y = Y[r_end - fit_size : r_end]
            
            coords = np.column_stack([fit_X, fit_Y])
            mean_coords = np.mean(coords, axis=0)
            centered = coords - mean_coords
            u, s, vh = np.linalg.svd(centered, full_matrices=False)
            line_dir = vh[0]
            
            for p_idx in range(r_end, next_end):
                pt = np.array([X[p_idx], Y[p_idx]])
                diff = pt - mean_coords
                proj = np.dot(diff, line_dir) * line_dir
                perp = diff - proj
                dist = np.linalg.norm(perp)
                
                pt_heading = headings[p_idx]
                cos_sim = abs(np.dot(pt_heading, line_dir))
                angle_diff = np.arccos(min(1.0, cos_sim)) * 180.0 / np.pi
                
                if dist < dist_threshold and angle_diff < angle_threshold:
                    states[p_idx] = 0
                else:
                    break
                    
    df_in['State_Final'] = states


router = APIRouter()


@router.post("/drone_analyze_xy")
async def drone_analyze_xy(file: UploadFile = File(...), sliding_window: str = Form("30")):
    print(f"Drone Analysis XY Request: {file.filename}, sliding_window: {sliding_window}")
    
    # 임시 디렉토리 생성하여 처리
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = os.path.join(temp_dir, file.filename)
        
        # 파일 저장
        content = await file.read()
        with open(temp_file_path, "wb") as f:
            f.write(content)
            
        target_file = temp_file_path
        print(f"Reading file: {target_file}")
        
        try:
            df_orig = pd.read_csv(target_file, header=None, dtype=str)
            df = df_orig.apply(pd.to_numeric, errors='coerce')
            
            # Timestamp 컬럼 찾기
            valid_start_idx = -1
            check_limit = min(df.shape[1], 5)
            
            for i in range(check_limit):
                try:
                    if float(df.iloc[0, i]) > 500000:
                        valid_start_idx = i
                        print(f"Timestamp column found at index {i}")
                        break
                except ValueError:
                    continue

            # 1. Time, Longitude, Latitude, Altitude 추출
            if valid_start_idx != -1 and df.shape[1] >= valid_start_idx + 4:
                df = df.iloc[:, [valid_start_idx, valid_start_idx+1, valid_start_idx+2, valid_start_idx+3]]
                df.columns = ['Time', 'Longitude', 'Latitude', 'Altitude']
            elif valid_start_idx != -1 and df.shape[1] >= valid_start_idx + 3:
                df = df.iloc[:, [valid_start_idx, valid_start_idx+1, valid_start_idx+2]]
                df.columns = ['Time', 'Longitude', 'Latitude']
                df['Altitude'] = 0.0
            else:
                print("Could not properly align columns. Defaulting to first 3 columns.")
                df = df.iloc[:, [0, 1, 2]]
                df.columns = ['Time', 'Longitude', 'Latitude']
                df['Altitude'] = 0.0

            # 결측치 및 문자열 에러 방지
            df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
            df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
            df['Altitude'] = pd.to_numeric(df['Altitude'], errors='coerce').fillna(0.0)
            df = df.dropna(subset=['Latitude', 'Longitude'])
            
            print("Data loaded. Applying Lon/Lat only segmentation...")

            # 2. X, Y 좌표만으로 비행 물리량 도출 및 전처리 (Notebook 로직 이식)
            R = 6371000.0

            # GPS 좌표 미세 노이즈 사전 평활화
            smooth_n = 5
            df['Latitude_s'] = df['Latitude'].rolling(smooth_n, center=True).mean().bfill().ffill()
            df['Longitude_s'] = df['Longitude'].rolling(smooth_n, center=True).mean().bfill().ffill()

            lat_rad = np.deg2rad(df['Latitude_s'])
            lon_rad = np.deg2rad(df['Longitude_s'])

            dx = R * lon_rad.diff().fillna(0) * np.cos(lat_rad.mean())
            dy = R * lat_rad.diff().fillna(0)
            dz = df['Altitude'].diff().fillna(0)  # 고도 변화량 (m)

            # 드론이 코너에서 멈칫할 때 발생하는 방향(Heading) 노이즈 제거
            ds = np.sqrt(dx**2 + dy**2 + dz**2)  # 3D 이동거리
            heading_rad = np.where(ds > 0.1, np.arctan2(dy, dx), np.nan)
            df['Pseudo_Yaw'] = np.rad2deg(heading_rad)
            df['Pseudo_Yaw'] = df['Pseudo_Yaw'].ffill().bfill() 

            yaw_rad_unwrapped = np.unwrap(np.deg2rad(df['Pseudo_Yaw']))
            df['Yaw_unwrap'] = np.rad2deg(yaw_rad_unwrapped)

            # 윈도우 사이즈 축소 (짧고 급격한 코너 포착)
            try:
                window_size = int(sliding_window) if sliding_window else 30
                if window_size <= 0:
                    window_size = 30
            except (ValueError, TypeError):
                window_size = 30
            df['Yaw_diff'] = df['Yaw_unwrap'].diff().fillna(0).abs()
            df['Yaw_rate'] = df['Yaw_diff'].rolling(window=window_size, center=True).mean().fillna(0)

            # 핵심 피처: 구간 내 최대 각도 변화량 (Max - Min)
            df['Yaw_range'] = (df['Yaw_unwrap'].rolling(window=window_size, center=True).max() - 
                               df['Yaw_unwrap'].rolling(window=window_size, center=True).min()).fillna(0)

            # 속도 피처: 구간 내 속도 변동성 (직선=낮음, 회전=높음)
            df['Speed'] = ds
            df['Speed_std'] = df['Speed'].rolling(window=window_size, center=True).std().fillna(0)

            # --- MATLAB 로직 포팅: 피처 계산 ---
            df['dx'] = dx
            df['dy'] = dy
            df['ds'] = ds

            # Helper functions for Tortuosity & Absolute Turn
            def local_tortuosity_multi(df_in, windows):
                n = len(df_in)
                dxv = df_in['dx'].values
                dyv = df_in['dy'].values
                dsv = df_in['ds'].values
                cdx = np.concatenate(([0], np.cumsum(dxv)))
                cdy = np.concatenate(([0], np.cumsum(dyv)))
                cds = np.concatenate(([0], np.cumsum(dsv)))
                tort = np.ones(n)
                for w in windows:
                    half = w // 2
                    for i in range(n):
                        i0 = max(0, i - half)
                        i1 = min(n - 1, i + half)
                        pl = cds[i1 + 1] - cds[i0]
                        nx = cdx[i1 + 1] - cdx[i0]
                        ny = cdy[i1 + 1] - cdy[i0]
                        nd = np.sqrt(nx**2 + ny**2)
                        if nd > 1.0:
                            t = pl / nd
                        elif pl > 3.0:
                            t = 5.0
                        else:
                            t = 1.0
                        if t > tort[i]:
                            tort[i] = t
                return tort

            def segment_tortuosity(dxv, dyv, dsv):
                pl = np.sum(dsv)
                nd = np.sqrt(np.sum(dxv)**2 + np.sum(dyv)**2)
                if nd > 0.5:
                    return pl / nd
                elif pl > 5.0:
                    return 10.0
                else:
                    return 1.0

            def windowed_abs_turn(yaw_unwrap, win):
                n = len(yaw_unwrap)
                yd = np.concatenate(([0], np.abs(np.diff(yaw_unwrap))))
                cyd = np.concatenate(([0], np.cumsum(yd)))
                at = np.zeros(n)
                half = win // 2
                for i in range(n):
                    i0 = max(0, i - half)
                    i1 = min(n - 1, i + half)
                    at[i] = cyd[i1 + 1] - cyd[i0]
                return at

            # Parameters
            tort_windows = [80, 120, 160]
            tort_thresh = 2.0
            fig8_turn_window = 200
            fig8_turn_thresh = 400.0
            fig8_tort_thresh = 2.5
            fig8_gap = 450
            merge_rotate_gap = 60
            min_seg_len = 20

            # Feature Engineering: Tortuosity & Absolute Turn
            df['Local_Tort'] = local_tortuosity_multi(df, tort_windows)
            df['AbsTurn'] = windowed_abs_turn(df['Yaw_unwrap'].values, fig8_turn_window)

            # 신규 피처: 곡률(Curvature) 계산 및 롤링 평균 적용
            dx_diff = df['dx'].diff().fillna(0)
            dy_diff = df['dy'].diff().fillna(0)
            ds_xy = np.sqrt(df['dx']**2 + df['dy']**2)
            curvature_raw = np.abs(df['dx'] * dy_diff - df['dy'] * dx_diff) / (ds_xy**3 + 1e-6)
            df['Curvature'] = curvature_raw.rolling(window=30, center=True).mean().fillna(0)

            # 신규 피처: 주기성(Periodicity) - ZCR 및 각속도 변동성 복합 메트릭
            yaw_diff_signed = df['Yaw_unwrap'].diff().fillna(0)
            # 노이즈를 필터링하기 위해 0.05도 이상 의미있는 변화만 감지
            zc = ((yaw_diff_signed > 0.05).astype(int).diff().fillna(0) != 0).astype(int)
            df['ZCR'] = zc.rolling(window=100, center=True).sum().fillna(0)

            yaw_mean = yaw_diff_signed.rolling(window=100, center=True).mean().fillna(0)
            yaw_std = yaw_diff_signed.rolling(window=100, center=True).std().fillna(0)
            df['Periodicity'] = df['ZCR'] * (yaw_std / (yaw_mean.abs() + 0.1))

            # 3. HMM Clustering
            feature_cols = ['Yaw_rate', 'Yaw_range', 'Local_Tort', 'AbsTurn', 'Curvature', 'Periodicity']
            df[feature_cols] = df[feature_cols].fillna(0)

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df[feature_cols])

            from hmmlearn import hmm
            print("Using Hierarchical GaussianHMM for segmentation...")
            # Step 1: Line vs Others
            model1 = hmm.GaussianHMM(n_components=2, covariance_type="diag", n_iter=100, random_state=42)
            model1.fit(X_scaled)
            state_raw_1 = model1.predict(X_scaled)

            # Find straight vs others (the state with the lower mean Local_Tort is straight)
            state_means_1 = {}
            for s in range(2):
                mask = (state_raw_1 == s)
                if np.sum(mask) > 0:
                    state_means_1[s] = np.mean(df.loc[mask, 'Local_Tort'])
                else:
                    state_means_1[s] = 0.0

            straight_label_1 = min(state_means_1.keys(), key=lambda s: state_means_1[s])
            others_label_1 = 1 - straight_label_1

            # Initialize State array (default all to 0: Line)
            state_arr = np.zeros(len(df), dtype=int)

            others_mask = (state_raw_1 == others_label_1)
            others_indices = np.where(others_mask)[0]

            if len(others_indices) > 0:
                if len(others_indices) < 5:
                    state_arr[others_indices] = 1
                else:
                    # Step 2: Others classified into Curve (1) vs Rotate (2)
                    scaler2 = StandardScaler()
                    X_scaled_others = scaler2.fit_transform(df.loc[others_mask, feature_cols])

                    # Get consecutive blocks of others to pass as lengths to HMM
                    split_indices = np.where(np.diff(others_indices) > 1)[0] + 1
                    blocks = np.split(others_indices, split_indices)
                    lengths = [len(b) for b in blocks]

                    model2 = hmm.GaussianHMM(n_components=2, covariance_type="diag", n_iter=100, random_state=42)
                    model2.fit(X_scaled_others, lengths=lengths)
                    state_raw_2 = model2.predict(X_scaled_others)

                    # Determine Curve (1) vs Rotate (2) based on AbsTurn
                    state_means_2 = {}
                    for s in range(2):
                        sub_mask = (state_raw_2 == s)
                        orig_indices_for_s = others_indices[sub_mask]
                        if len(orig_indices_for_s) > 0:
                            state_means_2[s] = np.mean(df.loc[orig_indices_for_s, 'AbsTurn'])
                        else:
                            state_means_2[s] = 0.0

                    rotate_label_2 = max(state_means_2.keys(), key=lambda s: state_means_2[s])
                    curve_label_2 = 1 - rotate_label_2

                    state_arr[others_indices[state_raw_2 == curve_label_2]] = 1
                    state_arr[others_indices[state_raw_2 == rotate_label_2]] = 2

            df['State'] = state_arr

            # HMM Turn OR Tortuosity hotspot -> Turn (to correct for slow turn/loops)
            hmm_state = df['State'].values
            tort_hotspot = (df['Local_Tort'] >= tort_thresh).values

            state_smooth = np.zeros(len(df), dtype=int)
            for i in range(len(df)):
                if hmm_state[i] == 2:
                    state_smooth[i] = 2
                elif hmm_state[i] == 1 or tort_hotspot[i]:
                    state_smooth[i] = 1
                else:
                    state_smooth[i] = 0

            df['State_Smooth'] = state_smooth

            # 연속적인 세그먼트 ID 생성을 위해 상태 변경 지점을 누적
            df['Segment_Change'] = (df['State_Smooth'].diff().fillna(0) != 0).astype(int)
            df['Final_Segment_ID'] = df['Segment_Change'].cumsum().astype(int)

            # =========================================================
            # [GHMM 성능 테스트 모드]
            # 4. PCA 기반 직선성 검증 및 5. 후처리 병합 필터 / 8자 라벨링 비활성화
            # -> GHMM (n_components=3) 의 분류 결과만을 그대로 시각화합니다.
            # =========================================================

            # # 4. PCA 기반 직선성 정밀 검증 (비활성화)
            # from sklearn.decomposition import PCA as _PCA
            # def is_linear_path(lats, lons, linearity_threshold=0.98): ...
            # def resegment(df_in): ...
            # rotate_seg_ids = df[df['State_Smooth'] == 1]['Final_Segment_ID'].unique()
            # for rid in rotate_seg_ids: ...  # PCA로 Rotate_Line / Rotate_Error 재분류

            # # 5. 체인 병합 알고리즘 (비활성화)
            # seg_order_df = ... ; merge_ops = ... ; n_merged = ...

            # # --- MATLAB 로직 포팅: 후처리 병합 필터 및 8자 라벨링 (비활성화) ---
            # def merge_adjacent_rotates(df_in, gap): ...
            # def absorb_short_segments(df_in, min_len, t_thresh): ...
            # def label_figure8(df_in, f8_turn_thresh, f8_tort_thresh, f8_gap): ...
            # if merge_rotate_gap > 0: merge_adjacent_rotates(df, merge_rotate_gap)
            # if min_seg_len > 0: absorb_short_segments(df, min_seg_len, tort_thresh)
            # label_figure8(df, fig8_turn_thresh, fig8_tort_thresh, fig8_gap)

            # GHMM State_Smooth 를 State_Final 로 직접 매핑
            # State_Smooth: 0=line(직선), 1=curve(커브), 2=rotate(8자/루프)
            df['State_Final'] = df['State_Smooth'].copy()
            # State_Smooth 2(HMM 8자)를 시각화 코드와 맞추기 위해 4로 매핑
            df.loc[df['State_Smooth'] == 2, 'State_Final'] = 4
            df['PCA_Ratio'] = 0.0  # PCA 비활성화: 0으로 초기화

            # [후처리 필터 적용]
            # 0. PCA 기반 곡선/회전 오감지 검증 (Curve Line Verification with PCA)
            # 임계값을 0.98로 높여 실제 회전 코너가 직선으로 오인되어 사라지는 것을 방지합니다.
            verify_curves_with_pca(df, linearity_threshold=0.98)

            # 2. PCA 기반 직선 경계 복원 및 병합 (PCA-based Boundary Refinement)
            # 기준을 타이트하게 변경하여 직선이 곡선 영역을 침범해 덮어쓰는 현상을 완화합니다.
            refine_line_boundaries(df, dist_threshold=1.0, angle_threshold=10.0)

            # resegment: State_Final 기준으로 Final_Segment_ID 재생성
            def resegment(df_in):
                sf = df_in['State_Final'].values
                changes = np.concatenate(([1], (sf[1:] != sf[:-1]).astype(int)))
                df_in['Final_Segment_ID'] = np.cumsum(changes).astype(int)

            resegment(df)

            # --- label_figure8 복원: GHMM rotate/curve 혼재 구간 통합 ---
            # AbsTurn(누적 회전량) + Local_Tort(굴곡도) 이 강하게 나타나는 구간 주변의
            # rotate(4) 및 curve(1) 세그먼트를 모두 rotate(4)로 통합합니다.
            def label_figure8(df_in, f8_turn_thresh, f8_tort_thresh, f8_gap):
                strong = (df_in['AbsTurn'] >= f8_turn_thresh) & (df_in['Local_Tort'] >= f8_tort_thresh)
                idx = np.where(strong)[0]
                if len(idx) == 0:
                    return

                groups = []
                gs = idx[0]
                ge = idx[0]
                for k in range(1, len(idx)):
                    if idx[k] - ge <= f8_gap:
                        ge = idx[k]
                    else:
                        groups.append((gs, ge))
                        gs = idx[k]
                        ge = idx[k]
                groups.append((gs, ge))

                n = len(df_in)
                for s0, s1 in groups:
                    # 주변의 rotate(4) 또는 curve(1) 구간까지 확장
                    while s0 > 0 and df_in.loc[df_in.index[s0 - 1], 'State_Final'] in (1, 4):
                        s0 -= 1
                    while s1 < n - 1 and df_in.loc[df_in.index[s1 + 1], 'State_Final'] in (1, 4):
                        s1 += 1
                    df_in.iloc[s0 : s1 + 1, df_in.columns.get_loc('State_Final')] = 4
                resegment(df_in)

            label_figure8(df, fig8_turn_thresh, fig8_tort_thresh, fig8_gap)
            print(f"[label_figure8] Applied: AbsTurn>={fig8_turn_thresh}, Local_Tort>={fig8_tort_thresh}, gap={fig8_gap}")


            # Segment ID 순번 재정렬
            alive_seg_ids = sorted(df[df['Final_Segment_ID'] != -1]['Final_Segment_ID'].unique())
            id_mapping = {old_id: new_idx for new_idx, old_id in enumerate(alive_seg_ids, start=1)}
            df.loc[df['Final_Segment_ID'] != -1, 'Final_Segment_ID'] = \
                df.loc[df['Final_Segment_ID'] != -1, 'Final_Segment_ID'].map(id_mapping).astype(int)

            # 6. 지도 시각화 및 다운로드 링크 생성
            output_file = os.path.join(temp_dir, "flight_path_segmented.html")
            output_split_dir = os.path.join(temp_dir, "splited")
            if not os.path.exists(output_split_dir):
                os.makedirs(output_split_dir)

            csv_links = []
            
            if 'Latitude' in df.columns and 'Longitude' in df.columns:
                center_lat = df['Latitude'].mean()
                center_lon = df['Longitude'].mean()
                import folium
                m = folium.Map(location=[center_lat, center_lon], zoom_start=16)
                
                all_coords = df[['Latitude', 'Longitude']].values.tolist()
                folium.PolyLine(
                    locations=all_coords, color="#000000", weight=5, opacity=0.5, tooltip="Original Path"
                ).add_to(m)
                
                # 36색 팔레트 적용
                colors = [
                    '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', '#FF8000', '#0080FF', 
                    '#FF0080', '#80FF00', '#00FF80', '#FF4040', '#4040FF', '#FFD700', '#ADFF2F', '#FF69B4', 
                    '#1E90FF', '#DC143C', '#8B4513', '#006400', '#4B0082', '#008080', '#D2691E', '#7FFFD4',
                    '#FF1493', '#32CD32', '#00008B', '#B8860B', '#800000', '#9ACD32', '#20B2AA', '#E9967A',
                    '#9400D3', '#FF6600', '#DA70D6', '#2E8B57'
                ]
                
                sorted_seg_ids = sorted(df[df['Final_Segment_ID'] != -1]['Final_Segment_ID'].unique())
                
                for seg_id in sorted_seg_ids:
                    group = df[df['Final_Segment_ID'] == seg_id]
                    coords = group[['Latitude', 'Longitude']].values.tolist()

                    seg_state_final = group['State_Final'].iloc[0] if 'State_Final' in group.columns else group['State_Smooth'].iloc[0]

                    if seg_state_final in (0, 2, 3):
                        type_str, label_prefix = "line", "line"
                        if seg_state_final == 0:
                            color = colors[(seg_id - 1) % len(colors)]
                            weight, opacity, dash_array = 3, 0.8, None
                            link_style = "display:block; margin:5px 0; color:#333;"
                        elif seg_state_final == 2:
                            color, weight, opacity, dash_array = '#9400D3', 5, 0.9, '10, 5'
                            link_style = "display:block; margin:5px 0; color:#9400D3; font-weight:bold;"
                        else:  # 3
                            color, weight, opacity, dash_array = '#FF6600', 5, 0.9, '10, 5'
                            link_style = "display:block; margin:5px 0; color:#FF6600; font-weight:bold;"
                    elif seg_state_final == 1:
                        type_str, label_prefix = "curve", "curve"
                        color = colors[(seg_id - 1) % len(colors)]
                        weight, opacity, dash_array = 5, 0.9, '5, 5'
                        link_style = "display:block; margin:5px 0; color:#0055aa; font-weight:bold;"
                    elif seg_state_final == 4:
                        type_str, label_prefix = "rotate", "rotate"
                        color, weight, opacity, dash_array = '#DC143C', 6, 0.95, None
                        link_style = "display:block; margin:5px 0; color:#DC143C; font-weight:bold;"
                    else:
                        type_str, label_prefix = f"State{seg_state_final}", f"State{seg_state_final}"
                        color, weight, opacity, dash_array = '#000000', 3, 0.8, None
                        link_style = "display:block; margin:5px 0; color:#000;"

                    label_text = f"{label_prefix}-{seg_id}"
                    if (seg_state_final != 0 )and 'PCA_Ratio' in group.columns:
                        pca_val = group['PCA_Ratio'].iloc[0]
                        label_text += f" (PCA: {pca_val:.3f})"

                    # 맵 시각화 추가
                    indices = group.index.values
                    if len(indices) > 0:
                        split_locs = np.where(np.diff(indices) > 1)[0] + 1
                        sub_groups_indices = np.split(indices, split_locs)
                        for sub_indices in sub_groups_indices:
                            if len(sub_indices) < 2: continue 
                            sub_coords = df.loc[sub_indices, ['Latitude', 'Longitude']].values.tolist()
                            folium.PolyLine(
                                locations=sub_coords, color=color, weight=weight, 
                                opacity=opacity, dash_array=dash_array, tooltip=f"ID {seg_id}: {type_str}"
                            ).add_to(m)
                            
                            folium.CircleMarker(
                                location=sub_coords[0], radius=3 if seg_state_final == 3 else 2,
                                color=color, fill=True, popup=f"ID {seg_id} [{type_str}] part start"
                            ).add_to(m)

                    # 마커 텍스트
                    center_lat_g, center_lon_g = group['Latitude'].mean(), group['Longitude'].mean()
                    if seg_state_final == 2:
                        label_html = f'<div style="font-size: 10pt; font-weight: bold; color: {color}; text-shadow: 1px 1px 2px white;">🔄 {label_text}</div>'
                    elif seg_state_final == 3:
                        label_html = f'<div style="font-size: 10pt; font-weight: bold; color: #FF6600; text-shadow: 1px 1px 2px white;">⚠️ {label_text}</div>'
                    elif seg_state_final == 4:
                        label_html = f'<div style="font-size: 10pt; font-weight: bold; color: #DC143C; text-shadow: 1px 1px 2px white;">♾️ {label_text}</div>'
                    else:
                        label_html = f'<div style="font-size: 10pt; font-weight: bold; color: {color}; text-shadow: 1px 1px 1px white;">{label_text}</div>'

                    folium.Marker(
                        location=[center_lat_g, center_lon_g],
                        icon=folium.DivIcon(icon_size=(180, 36), icon_anchor=(0, 0), html=label_html)
                    ).add_to(m)

                    # CSV 생성 및 다운로드 링크 (원본 데이터 + 분석 결과)
                    file_name = f"{label_prefix}-{seg_id}.csv"
                    save_path = os.path.join(output_split_dir, file_name)

                    # Generate CSV content (원본 데이터 + HMM/PCA/Color 컬럼 추가)
                    group_orig = df_orig.loc[group.index].copy()

                    # 분석 결과 컬럼 추가
                    group_orig['Yaw_rate'] = group['Yaw_rate'].values         # 윈도우 평균 각도 변화율 (deg/step)
                    group_orig['Yaw_range'] = group['Yaw_range'].values       # 윈도우 내 각도 스윙 (deg)
                    group_orig['Speed_std'] = group['Speed_std'].values       # 윈도우 내 속도 변동성 (m/step std)
                    group_orig['HMM_State'] = group['State'].values           # HMM 레이블 (0=Straight, 1=Turn)
                    group_orig['Local_Tort'] = group['Local_Tort'].values     # 국소 굴곡도
                    group_orig['AbsTurn'] = group['AbsTurn'].values           # 절대 누적 회전량
                    group_orig['PCA_Ratio'] = group['PCA_Ratio'].values       # PCA 설명 분산 비율
                    
                    # 최종 분류 컬럼명을 line, curve, rotate 텍스트로 치환하여 저장
                    flight_type_map = {0: 'line', 1: 'curve', 2: 'line', 3: 'line', 4: 'rotate'}
                    group_orig['State_Final'] = [flight_type_map.get(x, 'unknown') for x in group['State_Final'].values]
                    group_orig['Color'] = color                               # 지도 표시 색상 (Hex)

                    csv_content = group_orig.to_csv(index=False, header=True, float_format='%.12f')

                    # Save for record
                    with open(save_path, "w", encoding='utf-8') as f:
                        f.write(csv_content)

                    import base64
                    b64_csv = base64.b64encode(csv_content.encode('utf-8')).decode('utf-8')
                    prefix_icon = "⚠️ " if seg_state_final == 3 else ("🔄 " if seg_state_final == 2 else ("♾️ " if seg_state_final == 4 else ""))
                    download_link = f'<a href="data:text/csv;base64,{b64_csv}" download="{file_name}" style="{link_style}">{prefix_icon}Download {file_name}</a>'
                    csv_links.append(download_link)

                m.save(output_file)
                
                with open(output_file, 'r', encoding='utf-8') as f:
                    html_content = f.read()

                # --- [새 기능] Matplotlib을 이용한 단순화된 시각화 이미지 생성 ---
                img_html = ""
                try:
                    import matplotlib.pyplot as plt
                    import io
                    
                    # 색상 이름 매핑
                    hex_to_color_name = {
                        '#FF0000': 'Red', '#00FF00': 'Lime', '#0000FF': 'Blue',
                        '#FFFF00': 'Yellow', '#FF00FF': 'Magenta', '#00FFFF': 'Cyan',
                        '#FF8000': 'Orange', '#0080FF': 'Azure', '#FF0080': 'DeepPink',
                        '#80FF00': 'Chartreuse', '#00FF80': 'SpringGreen', '#FF4040': 'LightRed',
                        '#4040FF': 'LightBlue', '#FFD700': 'Gold', '#ADFF2F': 'GreenYellow',
                        '#FF69B4': 'HotPink', '#1E90FF': 'DodgerBlue', '#DC143C': 'Crimson',
                        '#9400D3': 'Violet', '#FF6600': 'OrangeRed',
                        '#008080': 'Teal', '#800000': 'Maroon', '#808000': 'Olive',
                        '#000080': 'Navy', '#32CD32': 'LimeGreen', '#FF7F50': 'Coral',
                        '#BA55D3': 'MediumOrchid', '#20B2AA': 'LightSeaGreen'
                    }
                    
                    cluster_color_map = {}
                    for seg_id in sorted_seg_ids:
                        group = df[df['Final_Segment_ID'] == seg_id]
                        seg_state_final = group['State_Final'].iloc[0] if 'State_Final' in group.columns else group['State_Smooth'].iloc[0]
                        
                        if seg_state_final == 0: color = colors[(seg_id - 1) % len(colors)]
                        elif seg_state_final == 1: color = colors[(seg_id - 1) % len(colors)]
                        elif seg_state_final == 2: color = '#9400D3'
                        elif seg_state_final == 3: color = '#FF6600'
                        elif seg_state_final == 4: color = '#DC143C'
                        else: color = '#000000'
                        
                        label_prefix = "Line" if seg_state_final == 0 else ("Rotate" if seg_state_final == 1 else ("Rotate_Line" if seg_state_final == 2 else ("Rotate_Error" if seg_state_final == 3 else ("Figure8" if seg_state_final == 4 else f"State{seg_state_final}"))))
                        full_label = f"{label_prefix}-{seg_id}"
                        c_name = hex_to_color_name.get(color, color)
                        if c_name not in cluster_color_map: cluster_color_map[c_name] = full_label
                        else: cluster_color_map[c_name] += f", {full_label}"

                    plt.figure(figsize=(10, 6))
                    plt.style.use('default')
                    plt.plot(df['Longitude'], df['Latitude'], color='#cccccc', linewidth=1, alpha=0.5)
                    
                    for seg_id in sorted_seg_ids:
                        group = df[df['Final_Segment_ID'] == seg_id]
                        seg_state_final = group['State_Final'].iloc[0] if 'State_Final' in group.columns else group['State_Smooth'].iloc[0]
                        
                        if seg_state_final == 0: c, lw, ls = colors[(seg_id - 1) % len(colors)], 2, '-'
                        elif seg_state_final == 1: c, lw, ls = colors[(seg_id - 1) % len(colors)], 3, '--'
                        elif seg_state_final == 2: c, lw, ls = '#9400D3', 3, ':'
                        elif seg_state_final == 3: c, lw, ls = '#FF6600', 3, '-.'
                        elif seg_state_final == 4: c, lw, ls = '#DC143C', 4, '-'
                        else: c, lw, ls = '#000000', 2, '-'

                        indices = group.index.values
                        if len(indices) > 0:
                            split_locs = np.where(np.diff(indices) > 1)[0] + 1
                            sub_groups_indices = np.split(indices, split_locs)
                            for sub_indices in sub_groups_indices:
                                if len(sub_indices) < 2: continue
                                sub_group = df.loc[sub_indices]
                                plt.plot(sub_group['Longitude'], sub_group['Latitude'], color=c, linewidth=lw, linestyle=ls)

                    plt.axis('off')
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
                    buf.seek(0)
                    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
                    plt.close()
                    
                    img_html = f"""
                    <div style="margin-top: 20px; border-top: 2px solid #eee; padding-top: 15px;">
                        <h4 style="margin:0 0 10px 0;">AI Analyze Overview</h4>
                        <img src="data:image/png;base64,{img_base64}" style="width:100%; height:auto; border: 1px solid #eee; border-radius: 5px;" />
                        <div style="font-size: 8.5pt; color: #333; margin-top: 10px; line-height: 1.4;">
                            <strong>Trajectory Styles:</strong><br/>
                            <span style="color:red;">━━</span> Solid: Line | <span style="color:blue;">---</span> Dash: Rotating<br/>
                            <span style="color:#9400D3;">...</span> Violet Dot: Rotate_Line | <span style="color:#FF6600;">-.-</span> Orange: Error<br/>
                            <span style="color:#DC143C;">━━</span> Crimson Solid: Figure8
                        </div>
                        <strong>Color Mapping for AI:</strong><br/>
                        <div style="font-size: 8pt; color: #555; margin-top: 10px; background: #f9f9f9; padding: 8px; border-radius: 5px; border: 1px solid #eee;">
                            <pre id="vlm_color_mapping" style="white-space: pre-wrap; margin:0;">{json.dumps(cluster_color_map, indent=1)}</pre>
                        </div>
                    </div>
                    """
                except Exception as plt_e:
                    print(f"Matplotlib error: {plt_e}")

                # --- 사이드바 레이아웃 구성 ---
                sidebar_html = f"""
                <div id="sidebar" style="position: fixed; top: 0; right: 0; width: 400px; height: 100vh; 
                            background-color: white; border-left: 2px solid #ccc; z-index: 10000; 
                            overflow-y: auto; padding: 20px; box-sizing: border-box; font-family: sans-serif; box-shadow: -2px 0 10px rgba(0,0,0,0.1);">
                    <h3 style="margin-top: 0; border-bottom: 2px solid #333; padding-bottom: 10px;">Analysis Report</h3>
                    
                    <div id="download_section">
                        <h4>Segment CSV Downloads</h4>
                        <div style="max-height: 300px; overflow-y: auto; border: 1px solid #eee; padding: 10px; border-radius: 5px;">
                            {''.join(csv_links)}
                        </div>
                    </div>

                    {img_html}
                </div>
                <style>
                    .folium-map {{ width: calc(100% - 400px) !important; height: 100vh !important; position: absolute !important; left: 0 !important; top: 0 !important; }}
                    @media (max-width: 1000px) {{
                        #sidebar {{ width: 100%; height: auto; position: relative; border-left: none; border-top: 2px solid #ccc; }}
                        .folium-map {{ width: 100% !important; height: 50vh !important; position: relative !important; }}
                    }}
                </style>
                """
                
                if "</body>" in html_content:
                    html_content = html_content.replace("</body>", f"{sidebar_html}</body>")
                else:
                    html_content += sidebar_html

                return HTMLResponse(content=html_content)
                
        except Exception as e:
            print(f"오류 발생: {e}")

