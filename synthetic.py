
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import random
import colorsys

import cv2
from alphabet import A, load_alphabet, char_token, sample_ngrams

space_token = char_token[A.Space]

@dataclass
class SynthSettings:

    spacing_multiplier: float = 0.9
    letter_size: tuple[int, int] = (50, 20) # Does nothign atm bcs it looked ugly
    image_size: tuple[int, int] = (400, 1000)

    margins: tuple[int, int] = (40, 40)
    allowed_portion_in_margin: float = 0.3
    line_space: int = 10


@dataclass
class Sample:
    tokens: list[int]
    image: np.ndarray
    segmentation: np.ndarray

def space_image():
    return np.full((40, 30), 255, dtype=np.uint8)

def _create_image(
        images: list[np.ndarray],
        char_tokens: list[int],
        settings: SynthSettings,
        reverse: bool = True
    ) -> Sample:

    # Init sizes
    image_size = settings.image_size
    image_height, image_width = image_size
    margin_vertical, margin_horizontal = settings.margins
    usable_height, usable_width = image_height - margin_vertical, image_width - margin_horizontal
    apim = settings.allowed_portion_in_margin
    leak_vertical, leak_horizontal = int(margin_vertical * apim), int(margin_horizontal * apim)
    max_char_height_per_row = 0

    start_x = image_size[1] - margin_horizontal
    cur_x = start_x
    cur_y = margin_vertical

    # Init images
    canvas = np.full((image_size[0], image_size[1]), 255, dtype=np.uint8)
    segmentation = np.full((len(char_token) - 1, image_size[0], image_size[1]), 0, dtype=np.uint8)

    iterator = list(zip(images, char_tokens))

    if reverse:
        iterator = reversed(iterator)

    consumed_letters = 0
    for letter_img, token in iterator:

        # letter_img = cv2.resize(letter_img, settings.letter_size)

        h, w = letter_img.shape[:2]

        left = cur_x - w
        out_of_bounds = left < 0
        out_of_leak = (left - margin_horizontal) < leak_horizontal
        if out_of_bounds or out_of_leak:
            # Next line
            cur_y += max_char_height_per_row + settings.line_space
            cur_x = start_x
            max_char_height_per_row = 0

        new_x = cur_x - w
        mask = letter_img < 200

        bottom = cur_y + h
        out_of_bounds = bottom >= image_height
        out_of_leak = (usable_height - bottom) < leak_vertical

        if out_of_bounds or out_of_leak:
            # Not enough space
            break

        consumed_letters += 1

        if token != space_token:
            segmentation[token, cur_y:bottom, new_x:cur_x] = mask.astype(np.uint8)

        canvas[cur_y:bottom, new_x:cur_x][mask] = letter_img[mask]

        cur_x = cur_x - int(w * settings.spacing_multiplier)
        max_char_height_per_row = max(max_char_height_per_row, h)


    if reverse:
        used_tokens = char_tokens[-consumed_letters:]
    else:
        used_tokens = char_tokens[:consumed_letters]

    return Sample(
        tokens=used_tokens,
        image=canvas,
        segmentation=segmentation
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
        char_tokens.append(char_token[letter])

    return _create_image(images, char_tokens, settings)


if __name__ == "__main__":

    from matplotlib import pyplot as plt

    alpha = load_alphabet()
    ngrams, _ = sample_ngrams(50)

    sequence = []

    for i, ngram in enumerate(ngrams):
        sequence.extend(ngram)

        if i != (len(ngrams) - 1):
            sequence.append(A.Space)

    sample = create_alphabet_image(
        sequence,
        # [A.Bet, A.Dalet, A.Mem, A.Qof, A.Space, A.Taw],
        alpha
    )

    fig, ax = plt.subplots()

    ax.imshow(sample.image, cmap="binary")

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


    # num_classes = sample.segmentation.shape[0]
    # colors = [
    #     tuple(int(c * 255) for c in colorsys.hsv_to_rgb(i / num_classes, 1.0, 1.0))
    #     for i in range(num_classes)
    # ]

    # overlay = cv2.cvtColor(sample.image, cv2.COLOR_GRAY2BGR)
    # for token, color in zip(range(num_classes), colors):
    #     mask = sample.segmentation[token].astype(bool)
    #     for c in range(3):
    #         overlay[..., c][mask] = color[c]

    # blended = overlay

    # fig, ax = plt.subplots(figsize=(12, 6))
    # ax.imshow(blended)
    # ax.set_title("Segmentation Overlay")
    # ax.axis('off')


    plt.show()


