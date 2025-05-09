import numpy as np
from skimage import measure, morphology, filters
from dataclasses import dataclass


@dataclass
class Box:
    minx:int
    maxx:int
    miny:int
    maxy:int

class LetterCentresExtractor:

    def __init__(self, min_size: int = 10, pad: int = 5):
        self.min_size = min_size
        self.pad = pad

   
    def __call__(self,
                 line_img: np.ndarray,
                 masks: np.ndarray) -> list[tuple[int, int]]:
       
        if masks.shape[0] != 27:
            raise ValueError("masks must have shape (27, H, W)")

        centres_all: list[tuple[int, int]] = []
        H, W = masks.shape[1:]

        for k in range(27):
            mask_k = masks[k].astype(bool)
            # remove tiny speckles
            mask_k = morphology.remove_small_objects(mask_k, self.min_size)

            lbl = measure.label(mask_k)
            props = measure.regionprops(lbl)

            centres_k: list[tuple[int, int]] = []
            for r in props:
                minr, minc, maxr, maxc = r.bbox
                # optional padding
                minr = max(minr - self.pad, 0)
                minc = max(minc - self.pad, 0)
                maxr = min(maxr + self.pad, H)
                maxc = min(maxc + self.pad, W)

                cx = (minc + maxc) // 2   # column  (x)
                cy = (minr + maxr) // 2   # row     (y)
                centres_k.append((cx, cy))

            centres_all.append(centres_k)

        return centres_all

    def _get_bounding_boxes(img: np.ndarray, self):
            
            gray = img.astype(float)
            blur = filters.gaussian(gray, sigma=1)
            thr = filters.threshold_otsu(blur)
            clean = morphology.remove_small_objects(blur < thr, min_size=30)
            lbls = measure.label(clean)
            raw_regions = measure.regionprops(lbls)

            regions = []
            for r in raw_regions:
                regions.append(r)

            boxes = []

            for r in regions:
                minr, minc, maxr, maxc = r.bbox[:4]
                minr = max(minr - self.pad, 0)
                minc = max(minc - self.pad, 0)
                maxr = min(maxr + self.pad, gray.shape[0])
                maxc = min(maxc + self.pad, gray.shape[1])
                boxes.append((minr, minc, maxr, maxc))

            return  boxes
    
    def _get_center(box:Box):

        cx = 