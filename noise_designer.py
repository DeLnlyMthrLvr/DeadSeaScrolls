import os
from pathlib import Path
from typing import Sequence
from matplotlib import pyplot as plt
import numpy as np
import pickle
import multiprocessing as mp

import tqdm

from synthetic import DataGenerator, SynthSettings, load_alphabet
from noise import Noise


def display_progression(
    n_progress: int = 3,
    n_variations: int = 5,
    perlin_strength: tuple[float, float] = (0, 0),
    warp_strength: tuple[float, float] = (0, 0),
    cutout_size: tuple[int, int] = (0, 0)
):

    downscale = 0.5

    print("Loading alphabet")
    alphabet = load_alphabet()

    ps = np.linspace(perlin_strength[0], perlin_strength[1], n_progress)
    ws = np.linspace(warp_strength[0], warp_strength[1], n_progress).round().astype(int)
    cs = np.linspace(cutout_size[0], cutout_size[1], n_progress).round().astype(int)

    normal = SynthSettings(downscale_factor=downscale)
    noise = Noise(normal.downscale_size)
    print("Perlin masks")
    noise.create_masks(N=20)

    print("Generations")
    fig, axs = plt.subplots(n_progress, n_variations, figsize=(4 * n_variations, 20))
    for i_var in range(n_variations):

        for i_row, (p, w, c) in enumerate(zip(ps, ws, cs, strict=True)):

            ax = axs[i_row, i_var]

            settings = SynthSettings(
                warp_noise=w > 0,
                warp_noise_strength=w,
                cutout_noise=c > 0,
                cutout_noise_size=c,
                downscale_factor=downscale
            )
            generator = DataGenerator(settings, alphabet)
            _, _, scrolls, _ = generator.generate_passages_scrolls(1)

            if p > 0:
                scrolls = noise.damage(scrolls, strength=p)

            ax.imshow(scrolls[0], cmap="binary")
            ax.set_axis_off()

    fig.tight_layout()
    plt.show()

def generate_data(
    n_progress: int = 3,
    n_variations: int = 500,
    perlin_strength: tuple[float, float] = (0.2, 0.25),
    warp_strength: tuple[float, float] = (0, 10),
    cutout_size: tuple[int, int] = (20, 120),
    batch_size: int = 250,
    n_noise_masks: int = 3
):

    data_folder = Path(__file__).parent / "data" / "scrolls"
    data_folder.mkdir(parents=True, exist_ok=True)

    downscale = 0.5
    n_batches_per_variation = n_variations // batch_size

    print("Loading alphabet")
    alphabet = load_alphabet()

    ps = np.linspace(perlin_strength[0], perlin_strength[1], n_progress)
    ws = np.linspace(warp_strength[0], warp_strength[1], n_progress).round().astype(int)
    cs = np.linspace(cutout_size[0], cutout_size[1], n_progress).round().astype(int)

    normal = SynthSettings(downscale_factor=downscale)
    noise = Noise(normal.downscale_size)
    print("Perlin masks")
    noise.create_masks(N=n_noise_masks)

    with tqdm.tqdm(total=n_batches_per_variation * n_progress) as pbar:

        for level, (p, w, c) in enumerate(zip(ps, ws, cs, strict=True)):

            noise_folder = data_folder / f"level_{level}"
            noise_folder.mkdir(parents=True, exist_ok=True)

            settings = SynthSettings(
                warp_noise=w > 0,
                warp_noise_strength=w,
                cutout_noise=c > 0,
                cutout_noise_size=c,
                downscale_factor=downscale
            )
            generator = DataGenerator(settings, alphabet)

            for i_batch in range(n_batches_per_variation):

                tokens, segmentation, scrolls, lines = generator.generate_passages_scrolls(batch_size, skip_char_seg=False)

                if p > 0:
                    scrolls = noise.damage(scrolls, strength=p)

                # Save one batch as noise_folder / f"chunk_{i_batch}.*", save Tokens, Scrolls, LineMasks do not use pickle for the numpy arrays
                name = f"chunk_{i_batch}"
                file_base = noise_folder
                with open(file_base / f"{name}.pickle", "wb") as f:
                    pickle.dump(tokens, f, protocol=pickle.HIGHEST_PROTOCOL)

                np.savez_compressed(
                    file_base / f"{name}.npz",
                    scrolls=scrolls.astype(np.uint8, copy=False),
                    line_masks=lines.astype(np.uint8, copy=False),
                    segmentation=segmentation.astype(np.uint8, copy=False)
                )

                pbar.update()


def _run_level(level_args: tuple):
    """Worker that generates all batches for one noise level.

    Parameters
    ----------
    level_args : tuple
        (level_idx, p_strength, w_strength, c_size,
         batch_size, n_batches_per_variation,
         downscale, n_noise_masks, data_folder)
    """

    (
        level,
        p,
        w,
        c,
        batch_size,
        n_batches_per_variation,
        downscale,
        n_noise_masks,
        data_folder,
    ) = level_args

    from synthetic import DataGenerator, SynthSettings, load_alphabet
    from noise import Noise

    alphabet = load_alphabet()

    settings = SynthSettings(
        warp_noise=w > 0,
        warp_noise_strength=w,
        cutout_noise=c > 0,
        cutout_noise_size=c,
        downscale_factor=downscale,
    )
    generator = DataGenerator(settings, alphabet)

    noise = Noise(settings.downscale_size)
    noise.create_masks(N=n_noise_masks)

    noise_folder = data_folder / f"level_{level}"
    noise_folder.mkdir(parents=True, exist_ok=True)

    for i_batch in range(n_batches_per_variation):
        tokens, segmentation, scrolls, lines = generator.generate_passages_scrolls(batch_size, skip_char_seg=False)

        if p > 0:
            scrolls = noise.damage(scrolls, strength=p)

        name = f"chunk_{i_batch}"
        with open(noise_folder / f"{name}.pickle", "wb") as f:
            pickle.dump(tokens, f, protocol=pickle.HIGHEST_PROTOCOL)

        np.savez_compressed(
            noise_folder / f"{name}.npz",
            scrolls=scrolls.astype(np.uint8, copy=False),
            line_masks=lines.astype(np.uint8, copy=False),
            segmentation=segmentation.astype(np.uint8, copy=False)
        )

        print(f"Level {level} – batch {i_batch + 1}/{n_batches_per_variation}")

def generate_data_mp(
    n_progress: int = 5,
    n_variations: int = 15_000,
    perlin_strength: tuple[float, float] = (0.2, 0.25),
    warp_strength: tuple[float, float] = (0, 10),
    cutout_size: tuple[int, int] = (20, 120),
    batch_size: int = 500,
    n_noise_masks: int = 30,
    n_workers: int = 5,
):

    mp.set_start_method("spawn", force=True)

    data_folder = Path(__file__).parent / "data" / "scrolls_seg"
    data_folder.mkdir(parents=True, exist_ok=True)

    downscale = 0.5
    n_batches_per_variation = n_variations // batch_size

    # Pre‑compute schedules for every noise level
    ps = np.linspace(perlin_strength[0], perlin_strength[1], n_progress)
    ws = np.linspace(warp_strength[0], warp_strength[1], n_progress).round().astype(int)
    cs = np.linspace(cutout_size[0], cutout_size[1], n_progress).round().astype(int)

    # Build argument tuples, one per future worker
    level_args = [
        (
            level,
            float(p),
            int(w),
            int(c),
            batch_size,
            n_batches_per_variation,
            downscale,
            n_noise_masks,
            data_folder,
        )
        for level, (p, w, c) in enumerate(zip(ps, ws, cs, strict=True))
    ]


    print(f"Spawning {n_workers} worker process(es)…")
    with mp.Pool(processes=n_workers) as pool:
        for _ in tqdm.tqdm(pool.imap_unordered(_run_level, level_args), total=n_progress):
            pass




# def _run_level(level_args: tuple):
#     """Worker that generates all batches for one noise level.

#     Parameters
#     ----------
#     level_args : tuple
#         (level_idx, p_strength, w_strength, c_size,
#          batch_size, n_batches_per_variation,
#          downscale, n_noise_masks, data_folder)
#     """
#     (level, p, w, c,
#      batch_size, n_batches_per_variation,
#      downscale, n_noise_masks, data_folder) = level_args

#     from synthetic import DataGenerator, SynthSettings, load_alphabet

#     alphabet = load_alphabet()

#     settings = SynthSettings(
#         warp_noise=w > 0,
#         warp_noise_strength=w,
#         cutout_noise=c > 0,
#         cutout_noise_size=c,
#         downscale_factor=downscale,
#     )
#     generator = DataGenerator(settings, alphabet)

#     noise = Noise(settings.downscale_size)
#     noise.create_masks(N=n_noise_masks)

#     noise_folder = data_folder / f"level_{level}"
#     noise_folder.mkdir(parents=True, exist_ok=True)

#     for i_batch in range(n_batches_per_variation):
#         tokens, _, scrolls, lines = generator.generate_passages_scrolls(batch_size)

#         if p > 0:
#             scrolls = noise.damage(scrolls, strength=p)

#         name = f"chunk_{i_batch}"
#         with open(noise_folder / f"{name}.pickle", "wb") as f:
#             pickle.dump(tokens, f, protocol=pickle.HIGHEST_PROTOCOL)

#         np.savez_compressed(
#             noise_folder / f"{name}.npz",
#             scrolls=scrolls.astype(np.uint8, copy=False),
#             line_masks=lines.astype(np.uint8, copy=False),
#         )

#         print(f"Level {level} {i_batch} / {n_batches_per_variation}")

# def generate_data(
#     n_progress: int = 6,
#     n_variations: int = 50_000,
#     perlin_strength: tuple[float, float] = (0.2, 0.25),
#     warp_strength: tuple[float, float] = (0, 10),
#     cutout_size: tuple[int, int] = (20, 120),
#     batch_size: int = 5_000,
#     n_noise_masks: int = 30,
#     n_workers: int | None = 3
# ):
#     """
#     Generate synthetic scroll data.

#     Multiprocessing strategy:
#       – One worker per *noise level* (outer loop in the original code).
#       – Each worker handles all its own batches sequentially.
#     """

#     mp.set_start_method("spawn", force=True)

#     data_folder = Path(__file__).parent / "data" / "scrolls"
#     data_folder.mkdir(parents=True, exist_ok=True)

#     downscale = 0.5
#     n_batches_per_variation = n_variations // batch_size

#     # Pre‑compute progress schedules
#     ps = np.linspace(perlin_strength[0], perlin_strength[1], n_progress)
#     ws = np.linspace(warp_strength[0], warp_strength[1], n_progress).round().astype(int)
#     cs = np.linspace(cutout_size[0], cutout_size[1], n_progress).round().astype(int)

#     # Build args for every worker
#     level_args = [
#         (
#             level,
#             float(p),
#             int(w),
#             int(c),
#             batch_size,
#             n_batches_per_variation,
#             downscale,
#             n_noise_masks,
#             data_folder,
#         )
#         for level, (p, w, c) in enumerate(zip(ps, ws, cs, strict=True))
#     ]

#     print(f"Spawning {n_workers} worker processes…")
#     with mp.Pool(processes=n_workers) as pool:
#         for _ in pool.imap_unordered(_run_level, level_args):
#             ...

def load_batches(level: int):
    level_path = Path(__file__).parent / "data" / "scrolls" / f"level_{level}"
    chunks = sorted(
        level_path.glob("chunk_*.npz"),
        key=lambda p: int(p.stem.split("_")[1])
    )

    for chunk_path in chunks:
        base = chunk_path.parent
        chunk = int(chunk_path.stem.split("_")[1])

        with open(base / f"chunk_{chunk}.pickle", "rb") as f:
            tokens: list[list[str]] = pickle.load(f) # not tokens but characters (batch, n_lines, sequence)

        data = np.load(chunk_path)
        scrolls: np.ndarray = data["scrolls"] # (batch, h, w)
        line_masks: np.ndarray = data["line_masks"] # (batch, h, w)
        yield tokens, scrolls, line_masks

if __name__ == "__main__":
    # display_progression(
    #     perlin_strength=(0.2, 0.25),
    #     warp_strength=(0, 10),
    #     cutout_size=(20, 120)
    # )

    # generate_data_mp()

    generate_data()

    # next(load_batches(0))