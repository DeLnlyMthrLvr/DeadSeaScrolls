import numpy as np
from dataclasses import dataclass
import cv2
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
@dataclass
class Box:
    minx: int
    maxx: int
    miny: int
    maxy: int

    def _get_area(self):
        return (self.maxx-self.minx) * (self.maxy-self.miny) 

class LetterCentresExtractor:
    def __init__(self, min_size: int = 50, pad: int = 5, sigma: float = 0.1):
        """
        min_size: minimum area (in pixels) for a region to keep
        pad: how many pixels to pad around each bounding box
        sigma: Gaussian blur sigma
        """
        self.min_size = min_size
        self.pad = pad
        self.sigma = sigma

    def __call__(self, masks: np.ndarray) -> list[tuple[int, int, int]]:
        """
        masks: numpy array of shape (27, H, W)
        Returns a list of (channel_index, center_row, center_col)
        """
        if masks.ndim != 3 or masks.shape[0] != 27:
            raise ValueError("masks must be a numpy array of shape (27, H, W)")

        centres_all = []
        _,H,W = masks.shape
        mask_area = H * W

        for chan_idx in range(masks.shape[0]):
            img = masks[chan_idx]
            boxes = self._get_bounding_boxes(img)
            for box in boxes:
                cy, cx = self._get_center(box)
                if (box._get_area() / mask_area) > 0.5:
                    continue
                centres_all.append((cx,cy, chan_idx))
        sorted_coords = sorted(centres_all,reverse=True, key=lambda x: x[0])
        return sorted_coords

    def _get_bounding_boxes(self, img: np.ndarray) -> list[Box]:
        # 1) blur
        blur = self._gaussian_blur(img, self.sigma)
        # 2) threshold
        thr = self._threshold_otsu(blur)
        # 3) binary mask
        bw = blur > thr
        # 4) label connected components
        labels, n_labels = self._fast_connected_components(bw)
        # 5) remove small objects
        labels = self._remove_small_objects(labels, n_labels, self.min_size)
        # 6) extract boxes
        boxes: list[Box] = []
        H, W = img.shape
        for lab in range(1, labels.max() + 1):
            ys, xs = np.where(labels == lab)
            if ys.size == 0:
                continue
            miny, maxy = ys.min(), ys.max() + 1
            minx, maxx = xs.min(), xs.max() + 1
            # pad
            miny = max(miny - self.pad, 0)
            minx = max(minx - self.pad, 0)
            maxy = min(maxy + self.pad, H)
            maxx = min(maxx + self.pad, W)
            boxes.append(Box(minx=int(minx), maxx=int(maxx), miny=int(miny), maxy=int(maxy)))
        return boxes

    def _get_center(self, box: Box) -> tuple[int, int]:
        """Return (row_center, col_center) of a Box."""
        cy = (box.miny + box.maxy) // 2
        cx = (box.minx + box.maxx) // 2
        return cy, cx


    def _gaussian_blur(self, img: np.ndarray, sigma: float) -> np.ndarray:
        return gaussian_filter(img, sigma=sigma, mode='constant', cval=0.0, truncate=3.0)

    def _threshold_otsu(self, img: np.ndarray) -> float:
        # compute 256-bin histogram
        hist, bin_edges = np.histogram(img.ravel(), bins=256)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        weight1 = np.cumsum(hist)
        weight2 = weight1[-1] - weight1
        mean1 = np.cumsum(hist * bin_centers) / np.maximum(weight1, 1)
        mean2 = (np.cumsum(hist * bin_centers)[-1] - np.cumsum(hist * bin_centers)) / np.maximum(weight2, 1)
        var_between = weight1 * weight2 * (mean1 - mean2) ** 2
        idx = np.argmax(var_between)
        return bin_centers[idx]

    def _fast_connected_components(self, bw:np.ndarray) -> tuple[np.ndarray, int]:
        mask8 = (bw > 0).astype(np.uint8) * 255

        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask8, connectivity=8)

        return labels, n_labels

    def _connected_components(self, bw: np.ndarray) -> tuple[np.ndarray, int]:
        """
        Two-pass (here simplified to flood-fill) connected component labelling.
        Returns (labels, n_labels).
        """
        H, W = bw.shape
        labels = np.zeros((H, W), dtype=int)
        current = 1
        for i in range(H):
            for j in range(W):
                if bw[i, j] and labels[i, j] == 0:
                    # flood fill from (i,j)
                    stack = [(i, j)]
                    labels[i, j] = current
                    while stack:
                        y, x = stack.pop()
                        for ny, nx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                            if 0 <= ny < H and 0 <= nx < W:
                                if bw[ny, nx] and labels[ny, nx] == 0:
                                    labels[ny, nx] = current
                                    stack.append((ny, nx))
                    current += 1
        return labels, current - 1

    def _remove_small_objects(self, labels: np.ndarray, n_labels: int, min_size: int) -> np.ndarray:
        """
        Zero out any connected component with fewer than min_size pixels.
        """
        # count pixels per label
        counts = np.bincount(labels.ravel())
        for lab in range(1, min(len(counts), n_labels + 1)):
            if counts[lab] < min_size:
                labels[labels == lab] = 0
        return labels
