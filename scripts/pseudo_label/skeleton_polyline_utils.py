#!/usr/bin/env python3
"""브러시로 그린 픽셀 마스크(두께가 프레임마다 들쭉날쭉함)를 skeleton화해서
중심선을 뽑고, 그 중심선을 단순화한 polyline으로 만든 뒤, 고정 두께로
다시 래스터화하는 유틸. TwinLiteNet+ 자체는 그대로 세그멘테이션 마스크로
학습하고, 이건 그 마스크(특히 ll)를 더 일관되게 만드는 전처리 단계다.
"""
from collections import deque

import cv2
import numpy as np
from skimage.morphology import skeletonize


def _neighbors(p, pts):
    y, x = p
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            q = (y + dy, x + dx)
            if q in pts:
                yield q


def _bfs_farthest(start, pts):
    visited = {start: None}
    q = deque([start])
    last = start
    while q:
        cur = q.popleft()
        last = cur
        for n in _neighbors(cur, pts):
            if n not in visited:
                visited[n] = cur
                q.append(n)
    return last, visited


def _order_skeleton_path(skel):
    """skeleton(2D bool) -> 한쪽 끝에서 반대쪽 끝까지 순서대로 이어진 (x,y) 점 리스트.
    가지(branch)가 있어도 가장 긴 경로(지름)를 택해서 짧은 곁가지는 무시함."""
    ys, xs = np.nonzero(skel)
    pts = set(zip(ys.tolist(), xs.tolist()))
    if not pts:
        return []
    if len(pts) == 1:
        y, x = next(iter(pts))
        return [(x, y)]
    a, _ = _bfs_farthest(next(iter(pts)), pts)
    b, visited = _bfs_farthest(a, pts)
    path = []
    cur = b
    while cur is not None:
        path.append(cur)
        cur = visited[cur]
    path.reverse()
    return [(x, y) for (y, x) in path]


def _simplify(points, epsilon):
    if len(points) < 3:
        return points
    arr = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    approx = cv2.approxPolyDP(arr, epsilon, closed=False)
    return [tuple(p[0].tolist()) for p in approx]


def mask_to_polylines(mask, min_area_frac=0.01, min_area_abs=15, epsilon=2.0):
    """이진 마스크 -> 컴포넌트별 중심선 polyline 리스트([[(x,y),...], ...]).
    면적이 작은 브러시 튐(noise speck)은 걸러낸다."""
    mask_u8 = (mask > 0).astype(np.uint8)
    total_area = int(mask_u8.sum())
    if total_area == 0:
        return []
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8)
    thresh = max(min_area_abs, min_area_frac * total_area)
    polylines = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < thresh:
            continue
        comp = labels == i
        skel = skeletonize(comp)
        path = _order_skeleton_path(skel)
        if len(path) < 2:
            continue
        simplified = _simplify(path, epsilon)
        if len(simplified) >= 2:
            polylines.append(simplified)
    return polylines


def polylines_to_mask(polylines, height, width, line_width=8):
    """polyline 리스트 -> 고정 두께로 래스터화한 이진(0/1) 마스크."""
    mask = np.zeros((height, width), dtype=np.uint8)
    for pts in polylines:
        arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(mask, [arr], isClosed=False, color=1, thickness=line_width, lineType=cv2.LINE_8)
    return mask
