import heapq
from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from skimage.morphology import disk, binary_dilation
import cv2

def find_local_maxima_2nd_derivative(arr):
    arr = np.asarray(arr)
    if arr.ndim != 1:
        raise ValueError("Input must be a 1D array.")


    first_deriv = np.gradient(arr)
    second_deriv = np.gradient(first_deriv)
    zero_crossings = np.where(np.diff(np.sign(first_deriv)) == -2)[0] + 1
    maxima_indices = [i for i in zero_crossings if second_deriv[i] < 0]

    return np.array(maxima_indices)

def label_hebrew_lines(seg: np.ndarray, min_gap: int = 5):

    H, W   = seg.shape
    labels = np.zeros_like(seg, dtype=np.int32)

    proj = seg.sum(axis=1)
    _n = 7
    proj = np.convolve(proj, np.ones(_n) / _n, mode="same")

    # peaks = find_local_maxima_2nd_derivative(proj)
    peaks, _ = find_peaks(proj)

    # fig, ax = plt.subplots()
    # vals = proj[peaks]
    # ax.bar(np.arange(len(proj)), proj)
    # ax.scatter(peaks, vals)


    # Merge close together
    to_merge = set(np.where(np.diff(peaks) < min_gap)[0])
    merged = []
    i = -1
    while i < len(peaks):
        i += 1

        if i >= len(peaks):
            break

        if i in to_merge:
            merged.append(int((peaks[i] + peaks[i + 1]) / 2))
            i += 1
            continue
        else:
            merged.append(peaks[i])


    peaks = np.array(merged)

    # For each peak, scan a small vertical window around it
    # peak_lines = seg[peaks, :]
    # plms = peak_lines > 0.5
    # starts = []
    # for line_id, (p, plm) in enumerate(zip(peaks, plms, strict=True)):
    #     starts.append((p, np.where(plm)[0].max(), line_id))

    starts = []
    for line_id, y0 in enumerate(peaks, start=1):
        y_min = max(0, y0 - 2)
        y_max = min(H, y0 + 2 + 1)
        # right‑most foreground pixel in that window
        cols = np.argmax(seg[y_min:y_max, ::-1], axis=1)
        best_offset = cols.min() # furthest to the right
        if best_offset == 0 and seg[y_min:y_max, -1].all() == 0:
            row = y0
            col = np.flatnonzero(seg[row])[-1]
        else:
            row_idx = np.where(cols == best_offset)[0][0] + y_min
            row = row_idx
            col = W - 1 - best_offset
        starts.append((row, col, line_id))

    pq = []
    for y, x, lid in starts:
        labels[y, x] = lid
        heapq.heappush(pq, (-x, lid, y, x))

    # neighbour order: left first, then up, then down
    neigh = [ ( 0, -1), (-1,  0), ( 1,  0) ]   # (dy, dx)

    while pq:
        negx, lid, y, x = heapq.heappop(pq)

        for dy, dx in neigh:
            ny, nx = y + dy, x + dx
            if (0 <= ny < H) and (0 <= nx < W):
                if seg[ny, nx] and labels[ny, nx] == 0:
                    labels[ny, nx] = lid
                    heapq.heappush(pq, (-nx, lid, ny, nx))

    return labels, peaks, proj, starts


def extract_lines_flood_fill(orignal_image: np.ndarray, mask: np.ndarray, inflate: int = 5) -> list[np.ndarray]:

    oh, ow = orignal_image.shape
    labels, _, _, _ = label_hebrew_lines(mask)

    # fig, ax = plt.subplots(1, 2)
    # ax[0].imshow(orignal_image)
    # ax[1].imshow(mask)

    selem = disk(inflate)

    line_imgs = []
    for group in np.unique(labels):

        if group == 0:
            continue

        gmask = labels == group
        inflated = binary_dilation(gmask, selem)

        # fig, ax = plt.subplots(1, 2)
        # ax[0].imshow(inflated, cmap="binary")
        inflated = cv2.resize(inflated.astype(np.float32), (ow, oh), interpolation=cv2.INTER_NEAREST)
        # ax[1].imshow(inflated, cmap="binary")

        inflated = inflated > 0.5

        ys, xs = np.nonzero(inflated)
        if ys.size == 0:
            continue

        xmin, xmax = ys.min(), ys.max()
        ymin, ymax = xs.min(), xs.max()

        if xmax - xmin <= 4 or ymax - ymin <= 20:
            continue

        sub_img  = orignal_image[xmin:xmax + 1, ymin:ymax + 1]
        sub_mask = inflated[xmin:xmax + 1, ymin:ymax + 1]

        line_img = sub_img.copy()
        line_img[~sub_mask] = 255

        line_imgs.append(line_img)

    return line_imgs
