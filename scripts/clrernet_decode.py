#!/usr/bin/env python3
"""CLRerNet ONNX(clrernet_no_nms_no_predictions_to_lanes_320x800.onnx)의 raw 출력을
CLRerNet/libs/models/dense_heads/clrernet_head.py의 get_lanes()/predictions_to_lanes()
로직을 그대로 numpy로 옮겨서 lane point list로 디코딩한다. NMS는 CUDA 전용 커스텀 연산이라
빼고, 대신 겹치는 lane을 평균 x-거리 기준으로 합치는 간단한 그리디 중복제거로 대체함
(우리는 벤치마크가 아니라 pseudo-label용 마스크만 필요해서 정밀한 CULane NMS까진 불필요).
"""
import numpy as np

NUM_PRIORS = 192
N_OFFSETS = 72
N_STRIPS = N_OFFSETS - 1
PRIOR_YS = np.linspace(1, 0, N_OFFSETS)  # index 0=화면 아래(가까움) -> index끝=화면 위(멈)


def softmax_np(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def predictions_to_lanes(xs, anchor_params, lengths, scores, ori_img_h, ori_img_w,
                          cut_height=0, extend_bottom=True):
    lanes = []
    for lane_xs, ap, length, score in zip(xs, anchor_params, lengths, scores):
        start = min(max(0, int(round((1 - ap[0]) * N_STRIPS))), N_STRIPS)
        length_i = int(round(length))
        end = min(start + length_i - 1, len(PRIOR_YS) - 1)
        if extend_bottom and start > 0:
            edge = (lane_xs[:start] >= 0.0) & (lane_xs[:start] <= 1.0)
            ext = 0
            for v in edge[::-1]:
                if v:
                    ext += 1
                else:
                    break
            start = max(0, start - ext)
        lane_ys = PRIOR_YS[start:end + 1]
        lane_xs_seg = lane_xs[start:end + 1]
        if len(lane_xs_seg) <= 1:
            continue
        lane_xs_seg = lane_xs_seg[::-1]
        lane_ys_seg = lane_ys[::-1]
        lane_ys_full = (lane_ys_seg * (ori_img_h - cut_height) + cut_height) / ori_img_h
        pts_x = lane_xs_seg * ori_img_w
        pts_y = lane_ys_full * ori_img_h
        lanes.append(list(zip(pts_x.tolist(), pts_y.tolist())))
    return lanes


def simple_dedup(lanes, x_dist_thresh=30):
    """비슷한 x 위치(중간 지점 기준)에 있는 lane들은 같은 차선으로 보고 하나만 남김.
    CUDA Line-NMS 대체용 간이 버전."""
    if not lanes:
        return lanes
    mid_xs = []
    for lane in lanes:
        xs = [p[0] for p in lane]
        mid_xs.append(xs[len(xs) // 2])
    order = np.argsort(mid_xs)
    kept = []
    kept_mid = []
    for i in order:
        mx = mid_xs[i]
        if all(abs(mx - km) > x_dist_thresh for km in kept_mid):
            kept.append(lanes[i])
            kept_mid.append(mx)
    return kept


def decode(cls_logits, anchor_params, lengths, xs, ori_img_h, ori_img_w,
           conf_threshold=0.41, cut_height=0):
    """
    cls_logits: (Np, 2), anchor_params: (Np, 3), lengths: (Np, 1) or (Np,), xs: (Np, Nr)
    -> list of lanes, each lane = list of (x, y) in 원본 이미지 좌표
    """
    scores = softmax_np(cls_logits, axis=-1)[:, 1]
    keep = scores >= conf_threshold
    if not np.any(keep):
        return [], scores
    xs_k = xs[keep]
    lengths_k = (lengths[keep, 0] if lengths.ndim == 2 else lengths[keep]) * N_STRIPS
    ap_k = anchor_params[keep]
    scores_k = scores[keep]

    lanes = predictions_to_lanes(xs_k, ap_k, lengths_k, scores_k, ori_img_h, ori_img_w, cut_height)
    lanes = simple_dedup(lanes)
    return lanes, scores_k
