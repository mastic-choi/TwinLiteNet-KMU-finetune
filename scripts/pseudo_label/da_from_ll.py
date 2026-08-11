#!/usr/bin/env python3
"""da(주행가능영역)는 항상 ll(차선) 최외곽 두 선 사이에 있다는 기하학적 관계를 이용해서
da를 만든다. GT로 확인해보니 각 row에서 da의 x범위가 그 row에 있는 ll 픽셀들의
min~max x와 거의 정확히 일치함(SAM처럼 "시각적 경계"를 찾는 게 아니라 애초에 ll이
정의하는 "의미적 경계"를 그대로 쓰는 방식이라 SAM의 근본적 한계를 우회함).

ll이 하나도 없는 row(카메라에 너무 가까워서 선이 화면 밖으로 나간 경우 등)는 전체 폭으로
채움 — GT에서도 그런 row는 대부분 꽉 차 있었음.
"""
import numpy as np


def geometric_da_from_ll(ll_mask):
    h, w = ll_mask.shape
    da = np.zeros((h, w), dtype=bool)
    for y in range(h):
        xs = np.where(ll_mask[y])[0]
        if len(xs) == 0:
            continue
        x0, x1 = xs.min(), xs.max()
        da[y, x0:x1 + 1] = True
    return da


def geometric_da_from_ll_with_fallback(ll_mask, roi_y0=250, roi_y1=390):
    """ROI 안에서 ll이 없는 row는 가장 가까운 유효 row의 범위를 이어받음(근처 row가
    비슷한 폭일 가능성이 높음) - 완전 빈 프레임이면 그대로 빈 da."""
    h, w = ll_mask.shape
    bounds = [None] * h
    for y in range(h):
        xs = np.where(ll_mask[y])[0]
        if len(xs) > 0:
            bounds[y] = (int(xs.min()), int(xs.max()))

    # forward-fill, then backward-fill
    last = None
    for y in range(h):
        if bounds[y] is not None:
            last = bounds[y]
        elif last is not None:
            bounds[y] = last
    last = None
    for y in range(h - 1, -1, -1):
        if bounds[y] is not None:
            last = bounds[y]
        elif last is not None:
            bounds[y] = last

    da = np.zeros((h, w), dtype=bool)
    for y in range(h):
        if bounds[y] is not None:
            x0, x1 = bounds[y]
            da[y, x0:x1 + 1] = True
    return da
