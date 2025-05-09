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
                 masks: list[np.ndarray]) -> list[tuple[int, int]]:
       
        if masks.shape[0] != 27:
            raise ValueError("masks must have shape (27, H, W)")

        centres_all: list[tuple[int, int]] = []

        for index, channel in enumerate(masks):
            boxes = self._get_bounding_boxes(channel)
            for box in boxes:
                center = self._get_center(box)
                centres_all.append((center[0], index))

        return centres_all

    def _get_bounding_boxes(img: np.ndarray, self) -> Box:
            
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
    
    def _get_center(box:Box) ->tuple[int,int]:
        cx = (box.minx + box.maxx) // 2
        cy = (box.miny + box.maxy) // 2
        return (cx, cy)