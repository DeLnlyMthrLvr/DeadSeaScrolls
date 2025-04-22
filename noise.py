

import numpy as np


def generate_perlin_noise_2d(shape, res, octaves=1, persistence=0.5, lacunarity=2.0, rng=None):
    if rng is None:
        rng = np.random.RandomState()

    def fade(t):
        return 6*t**5 - 15*t**4 + 10*t**3

    def perlin(shape, res):
        y = np.linspace(0, res[0], shape[0], endpoint=False)
        x = np.linspace(0, res[1], shape[1], endpoint=False)
        gy, gx = np.meshgrid(y, x, indexing='ij')
        iy = gy.astype(int)
        ix = gx.astype(int)
        ty = gy - iy
        tx = gx - ix
        u = fade(tx)
        v = fade(ty)
        angles = 2 * np.pi * rng.rand(res[0] + 1, res[1] + 1)
        grads = np.dstack((np.cos(angles), np.sin(angles)))
        g00 = grads[iy, ix]
        g10 = grads[iy, ix + 1]
        g01 = grads[iy + 1, ix]
        g11 = grads[iy + 1, ix + 1]
        d00 = np.stack((tx,     ty    ), axis=-1)
        d10 = np.stack((tx - 1, ty    ), axis=-1)
        d01 = np.stack((tx,     ty - 1), axis=-1)
        d11 = np.stack((tx - 1, ty - 1), axis=-1)
        n00 = np.sum(g00 * d00, axis=-1)
        n10 = np.sum(g10 * d10, axis=-1)
        n01 = np.sum(g01 * d01, axis=-1)
        n11 = np.sum(g11 * d11, axis=-1)
        nx0 = n00 + u * (n10 - n00)
        nx1 = n01 + u * (n11 - n01)
        return nx0 + v * (nx1 - nx0)

    noise = np.zeros(shape)
    frequency = 1
    amplitude = 1
    max_amp = 0

    for _ in range(octaves):
        res_i = (int(res[0] * frequency), int(res[1] * frequency))
        noise += perlin(shape, res_i) * amplitude
        max_amp += amplitude
        amplitude *= persistence
        frequency *= lacunarity

    return noise / max_amp


class Noise:

    def __init__(self, image_size: tuple[int, int], allow_noise_augmentation: bool = True):
        self.perlin: np.ndarray = None

        self.image_size = image_size
        self.allow_noise_augmentation = allow_noise_augmentation

    def create_masks(self, N: int = 10):
        """Expensive operation call only few times during training, potentially only once
        """

        perlin = []
        for _ in range(N):
            noise = generate_perlin_noise_2d(self.image_size, (8, 8), octaves=8, persistence=0.7, lacunarity=2)
            perlin.append(noise)

            if self.allow_noise_augmentation:
                perlin.append(np.fliplr(noise))
                perlin.append(np.rot90(noise, k=2))


        self.perlin = np.stack(perlin, axis=0)


    def damage(self, images: np.ndarray, strength: float = 0.2):
        """Cheaper operation
        """

        images = images.copy()

        assert self.perlin is not None

        t = -0.3 + strength

        N = images.shape[0]
        M = self.perlin.shape[0]
        idx = np.random.randint(0, M, size=N)

        batch_masks = self.perlin[idx, :, :]
        mask = batch_masks < t
        images[mask] = 255

        return images


