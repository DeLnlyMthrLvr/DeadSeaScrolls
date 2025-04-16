
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import random

import cv2
from alphabet import A, load_alphabet, char_token, sample_ngrams

space_token = char_token[A.Space]

@dataclass
class SynthSettings:

    spacing_multiplier: float = 0.9
    offset: tuple[int, int] = (0, 0)
    letter_size: tuple[int, int] = (50, 20)


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
        image_size: tuple[int, int],
        settings: SynthSettings
    ) -> Sample:

    canvas = np.full((image_size[1], image_size[0]), 255, dtype=np.uint8)
    segmentation = np.full((len(char_token) - 1, image_size[1], image_size[0]), 0, dtype=np.uint8)

    cur_x = image_size[0] + settings.offset[0]
    cur_y = settings.offset[1]

    row_height = 0
    for letter_img, token in reversed(list(zip(images, char_tokens))):

        # letter_img = cv2.resize(letter_img, settings.letter_size)

        h, w = letter_img.shape[:2]
        if cur_x - w < 0:
            cur_y += row_height

            cur_x = image_size[0]
            row_height = 0
        new_x = cur_x - w

        mask = letter_img < 200

        if token != space_token:
            segmentation[token, cur_y:cur_y+h, new_x:cur_x] = mask.astype(np.uint8)

        canvas[cur_y:cur_y+h, new_x:cur_x][mask] = letter_img[mask]

        cur_x = cur_x - int(w * settings.spacing_multiplier)
        row_height = max(row_height, h)

    return Sample(
        tokens=char_tokens,
        image=canvas,
        segmentation=segmentation
    )

def create_alphabet_image(
        text: list[A],
        image_size: tuple[int, int],
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

    return _create_image(images, char_tokens, image_size, settings)


if __name__ == "__main__":

    from matplotlib import pyplot as plt

    alpha = load_alphabet()
    ngrams, _ = sample_ngrams(30)

    sequence = []

    for i, ngram in enumerate(ngrams):
        sequence.extend(ngram)

        if i != (len(ngrams) - 1):
            sequence.append(A.Space)

    sample = create_alphabet_image(
        sequence,
        # [A.Bet, A.Dalet, A.Mem, A.Qof, A.Space, A.Taw],
        (1000, 400),
        alpha
    )

    plt.imshow(sample.image, cmap="Greys")
    plt.show()


