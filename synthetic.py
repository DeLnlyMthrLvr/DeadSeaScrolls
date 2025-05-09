
from dataclasses import dataclass
import numpy as np
import random

import cv2
import tqdm
from alphabet import A, load_alphabet, enum_to_hebrew, sample_ngrams, load_n_grams, MEAN_NGRAM_CHAR, MEAN_CHAR_HEIGHT, MEAN_CHAR_WIDTH
from noise import Noise, cutout_noise, warp_mask
from bible import BibleTexts


@dataclass
class SynthSettings:

    spacing_multiplier: float = 0.9
    image_size: tuple[int, int] = (676, 902)
    downscale_factor: float = 1
    downscale_size: tuple[int, int] = None

    margins: tuple[int, int] = (40, 40)
    allowed_portion_in_margin: float = 0.3
    line_space: int = 5
    line_seg_offset: int = 12

    cutout_noise: bool = False
    cutout_noise_size: int = 100

    warp_noise: bool = False
    warp_noise_strength: float = 8

    def __post_init__(self):
        self.downscale_size = (int(self.image_size[0] * self.downscale_factor), int(self.image_size[1] * self.downscale_factor))


@dataclass
class Sample:
    characters: list[list[str]]
    image: np.ndarray
    segmentation: np.ndarray
    line: np.ndarray

def space_image():
    return np.full((40, 30), 255, dtype=np.uint8)

def _create_image(
        images: list[np.ndarray],
        enums: list[A],
        settings: SynthSettings
    ) -> Sample:

    # Init sizes
    image_size = settings.image_size
    image_height, image_width = image_size
    margin_vertical, margin_horizontal = settings.margins
    usable_height, usable_width = image_height - margin_vertical, image_width - margin_horizontal
    apim = settings.allowed_portion_in_margin
    leak_vertical, leak_horizontal = int(margin_vertical * apim), int(margin_horizontal * apim)
    max_char_height_per_row = 0
    line_seg_offset = settings.line_seg_offset

    start_x = image_size[1] - margin_horizontal
    cur_x = start_x
    cur_y = margin_vertical

    # Init images
    canvas = np.full((image_size[0], image_size[1]), 255, dtype=np.uint8)
    segmentation = np.full((len(enum_to_hebrew), image_size[0], image_size[1]), 0, dtype=np.uint8)
    line = np.full((image_size[0], image_size[1]), 0, dtype=np.uint8)


    # Line data
    line_max_x: float = -float("inf")
    line_min_x: float = float("inf")
    line_max_y: float = -float("inf")
    line_min_y: float = float("inf")

    cnoise = settings.cutout_noise
    if cnoise:
        cutout_mask = cutout_noise(*settings.image_size, radius=settings.cutout_noise_size)

    iterator = list(zip(images, enums))
    used_chars_split_to_lines: list[str] = []
    line_chars: list[str] = []

    def line_end():
        nonlocal line_chars, used_chars_split_to_lines, line_max_x, line_min_x, line_max_y, line_min_y

        if len(line_chars) > 0:
            used_chars_split_to_lines.append(''.join(line_chars))
            line_chars = []

            if not (line_max_x == -float("inf") or line_min_x == float("inf") or line_max_y == -float("inf") or line_min_y == float("inf")):
                line[
                    int(line_min_y) + line_seg_offset : int(line_max_y) - line_seg_offset,
                    int(line_min_x) - 2: int(line_max_x) + 2
                ] = 1

        line_max_x = -float("inf")
        line_min_x = float("inf")
        line_max_y = -float("inf")
        line_min_y = float("inf")

    i = -1
    while i < len(iterator):
        i+= 1

        letter_img, enum = iterator[i]

        h, w = letter_img.shape[:2]

        left = cur_x - w
        out_of_bounds = left < 0
        out_of_leak = (left - margin_horizontal) < leak_horizontal
        if out_of_bounds or out_of_leak:
            # Next line
            cur_y += max_char_height_per_row + settings.line_space
            cur_x = start_x
            max_char_height_per_row = 0
            line_end()

        new_x = cur_x - w
        mask = letter_img < 200

        bottom = cur_y + h
        out_of_bounds = bottom >= image_height
        out_of_leak = (usable_height - bottom) < leak_vertical

        if out_of_bounds or out_of_leak:
            # Not enough space
            line_end()
            break

        # Check if it collides with cutout mask
        if cnoise and (enum != A.Space):
            cmask = cutout_mask[cur_y:bottom, new_x:cur_x]
            letter_mask = letter_img < 200
            letter_pixels = letter_mask.sum()

            assert letter_pixels > 0

            p_collision = (letter_mask & cmask).astype(float).sum() / letter_pixels

            if p_collision > 0.5:
                # Try again further
                cur_x -= int(w / 0.5)
                i -= 1
                continue

        # Letter can be applied
        if enum != A.Space:

            # segmentation[enum, cur_y:bottom, new_x:cur_x] = mask.astype(np.uint8)

            line_chars.append(enum_to_hebrew[enum])

            line_min_y = min(cur_y, line_min_y)
            line_min_x = min(new_x, line_min_x)
            line_max_y = max(bottom, line_max_y)
            line_max_x = max(cur_x, line_max_x)

        canvas[cur_y:bottom, new_x:cur_x][mask] = letter_img[mask]

        cur_x = cur_x - int(w * settings.spacing_multiplier)
        max_char_height_per_row = max(max_char_height_per_row, h)

    if settings.warp_noise:
        canvas, line = warp_mask(canvas, line, warp_strength=settings.warp_noise_strength)

    if settings.downscale_factor < 1.0:
        sd_height, sd_width = settings.downscale_size
        canvas = cv2.resize(canvas, (sd_width, sd_height), interpolation=cv2.INTER_AREA)

        res_seg = np.empty((segmentation.shape[0], sd_height, sd_width), dtype=np.uint8)
        for i_seg in range(segmentation.shape[0]):
            res_seg[i_seg, ...] = cv2.resize(segmentation[i_seg, ...], (sd_width, sd_height), interpolation=cv2.INTER_AREA)
        segmentation = res_seg

        line = cv2.resize(line, (sd_width, sd_height), interpolation=cv2.INTER_AREA)

    return Sample(
        characters=used_chars_split_to_lines,
        image=canvas,
        segmentation=segmentation,
        line=line
    )

def create_alphabet_image(
        text: list[A],
        alphabet: dict[str, list[np.ndarray]],
        settings: SynthSettings | None = None
    ) -> Sample:

    if settings is None:
        settings = SynthSettings()

    images = []
    char_tokens = []
    for letter in text:
        key = letter.value

        if letter == A.Space:
            images.append(space_image())
        else:
            letter_img = random.choice(alphabet[key])
            images.append(letter_img)
        char_tokens.append(letter)

    return _create_image(images, char_tokens, settings)


TokensPerLine = list[list[str]] # (n, n_lines, sequence_length)
SegmentationMasks = np.ndarray # (n, char_chanels, height, width)
ScrollImages = np.ndarray # (n, height, width)
LineMasks = np.ndarray # ()

class DataGenerator:

    def __init__(
        self,
        settings: SynthSettings | None = None,
        alphabet = None
    ):

        h, w = settings.image_size
        tpl = w / MEAN_CHAR_WIDTH
        tpc = h / MEAN_CHAR_HEIGHT

        tps = round(tpl * tpc)

        if alphabet is None:
            self.alphabet = load_alphabet()
        else:
            self.alphabet = alphabet

        self.ngrams, self.ngram_frequencies, self.ngram_tokens = load_n_grams()
        self.settings = SynthSettings() if settings is None else settings
        self.max_sequence_length = tps + 3
        self.gen_ngrams = int((self.max_sequence_length // MEAN_NGRAM_CHAR) + 10)
        self.bible = BibleTexts(self.max_sequence_length)

    def generate_ngram_scrolls(self, N: int = 1_000, skip_char_seg: bool = True) -> tuple[TokensPerLine, SegmentationMasks, ScrollImages, LineMasks]:

        batch_characters = []
        batch_seg_masks = []
        batch_scrolls = []
        batch_lines = []

        for _ in tqdm.tqdm(range(N), total=N, disable=True):
            ngrams, _ = sample_ngrams(self.gen_ngrams, self.ngrams, self.ngram_frequencies, self.ngram_tokens)

            sequence = []

            for i, ngram in enumerate(ngrams):
                sequence.extend(ngram)

                if len(sequence) >= self.max_sequence_length:
                    break

                # Add space between ngrams
                if i != (len(ngrams) - 1):
                    sequence.append(A.Space)

            sequence = sequence[:self.max_sequence_length]

            sample = create_alphabet_image(
                sequence,
                self.alphabet,
                self.settings
            )

            if not skip_char_seg:
                batch_seg_masks.append(sample.segmentation[np.newaxis, ...])

            batch_characters.append(sample.characters)
            batch_scrolls.append(sample.image[np.newaxis, ...])
            batch_lines.append(sample.line[np.newaxis, ...])

        return (
            batch_characters,
            np.empty((N, 0)) if skip_char_seg else np.concat(batch_seg_masks, axis=0),
            np.concat(batch_scrolls, axis=0),
            np.concat(batch_lines, axis=0)
        )


    def generate_passages_scrolls(self, N: int = 1_000, skip_char_seg: bool = True) -> tuple[TokensPerLine, SegmentationMasks, ScrollImages, LineMasks]:

        batch_characters = []
        batch_seg_masks = []
        batch_scrolls = []
        batch_lines = []

        for passage in self.bible.sample_passages(N):

            sample = create_alphabet_image(
                passage,
                self.alphabet,
                self.settings
            )

            if not skip_char_seg:
                batch_seg_masks.append(sample.segmentation[np.newaxis, ...])

            batch_characters.append(sample.characters)
            batch_scrolls.append(sample.image[np.newaxis, ...])
            batch_lines.append(sample.line[np.newaxis, ...])

        return (
            batch_characters,
            np.empty((N, 0)) if skip_char_seg else np.concat(batch_seg_masks, axis=0),
            np.concat(batch_scrolls, axis=0),
            np.concat(batch_lines, axis=0)
        )


def extract_lines_cc(
        img: np.ndarray,
        binary_mask: np.ndarray,
        min_area: int = 500,
        inflate: int = 6
    ) -> list[np.ndarray]:

    mask8 = (binary_mask > 0).astype(np.uint8) * 255

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask8, connectivity=8)

    h, w = binary_mask.shape
    lines = []
    for lab in range(1, n_labels):
        x, y, bw, bh, area = stats[lab]
        if area < min_area:
            continue

        x0 = max(x - inflate, 0)
        y0 = max(y - inflate, 0)
        x1 = min(x + bw + inflate, w)
        y1 = min(y + bh + inflate, h)

        crop = img[y0:y1, x0:x1].copy()
        lines.append(crop)

    return lines

if __name__ == "__main__":

    from matplotlib import pyplot as plt

    generator = DataGenerator(settings=SynthSettings(downscale_factor=1))
    noise = Noise(generator.settings.downscale_size)

    tokens, seg, scrolls, lines = generator.generate_passages_scrolls(10, skip_char_seg=True)

    print(len(tokens), len(tokens[0]))
    print(tokens)

    # noise.create_masks(2)
    # dmgd = noise.damage(scrolls, strength=0.3)

    # print("Scrolls", len(tokens))
    # print("Lines", len(tokens[0]), len(tokens[1]))
    # print("Chars", len(tokens[0][0]), len(tokens[1][1]))


    # for i in range(scrolls.shape[0]):
    #     fig, ax = plt.subplots(1, 2)

    #     ax[0].imshow(scrolls[i], cmap="binary")
    #     ax[1].imshow(lines[i], cmap="binary_r")

    #     fig.tight_layout()

        # img_lines = extract_lines_cc(scrolls[i], lines[i])

        # fig, axs = plt.subplots(len(img_lines), 1)

        # if not isinstance(axs, np.ndarray):
        #     axs = np.array([axs], dtype=object)

        # for ax, img_line in zip(axs.ravel(), img_lines, strict=True):
        #     ax.imshow(img_line, cmap="binary")



    # masks = sample.segmentation
    # num_masks = masks.shape[0]
    # masks_per_fig = 9

    # figs = []
    # axs_all = []
    # for i in range(0, num_masks, masks_per_fig):
    #     fig, axs = plt.subplots(3, 3, figsize=(9, 9))
    #     axs = axs.flatten()
    #     for j in range(min(masks_per_fig, num_masks - i)):
    #         axs[j].imshow(masks[i + j], cmap="Greys")

    #     figs.append(fig)
    #     axs_all.append(axs)


    # plt.show()


