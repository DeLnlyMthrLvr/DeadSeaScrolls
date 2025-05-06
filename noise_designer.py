from matplotlib import pyplot as plt
import numpy as np

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



# def display_progression(
#     n_progress: int = 10,
#     batch_size: int = 5_000,
#     batches: int = 5,
#     perlin_strength: tuple[float, float] = (0, 0),
#     warp_strength: tuple[float, float] = (0, 0),
#     cutout_size: tuple[int, int] = (0, 0)
# ):

#     downscale = 0.5

#     print("Loading alphabet")
#     alphabet = load_alphabet()

#     ps = np.linspace(perlin_strength[0], perlin_strength[1], n_progress)
#     ws = np.linspace(warp_strength[0], warp_strength[1], n_progress).round().astype(int)
#     cs = np.linspace(cutout_size[0], cutout_size[1], n_progress).round().astype(int)

#     normal = SynthSettings(downscale_factor=downscale)
#     noise = Noise(normal.downscale_size)
#     print("Perlin masks")
#     noise.create_masks(N=20)

#     print("Generations")
#     fig, axs = plt.subplots(n_progress, n_batch, figsize=(4 * n_batch, 20))
#     for i_var in range(n_batch):

#         for i_row, (p, w, c) in enumerate(zip(ps, ws, cs, strict=True)):

#             ax = axs[i_row, i_var]

#             settings = SynthSettings(
#                 warp_noise=w > 0,
#                 warp_noise_strength=w,
#                 cutout_noise=c > 0,
#                 cutout_noise_size=c,
#                 downscale_factor=downscale
#             )
#             generator = DataGenerator(settings, alphabet)
#             _, _, scrolls, _ = generator.generate_passages_scrolls(1)

#             if p > 0:
#                 scrolls = noise.damage(scrolls, strength=p)

#             ax.imshow(scrolls[0], cmap="binary")
#             ax.set_axis_off()

#     fig.tight_layout()
#     plt.show()


if __name__ == "__main__":
    display_progression(
        perlin_strength=(0.2, 0.25),
        warp_strength=(0, 10),
        cutout_size=(20, 120)
    )