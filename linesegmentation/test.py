from pathlib import Path
from PIL import Image

from matplotlib import pyplot as plt
from unet import LineSegmenter
from pathlib import Path
import numpy as np


run_name = "20250507_203729_wide_unet_largek_fixed_final"
experiment_folder = Path(__file__).parent / "runs" / run_name
model = LineSegmenter.load(experiment_folder)
model = model.to("cuda")

test_data_path = Path(__file__).parent.parent / "data" / "image-data"

test_imgs = []

for ent in test_data_path.iterdir():

    if ent.is_dir() or not ent.name.endswith("binarized.jpg"):
        continue

    img = Image.open(ent).convert("L")
    img_array = np.array(img)
    test_imgs.append(img_array)

    break

out = model.process_heterogenous_images(test_imgs)

rand_i = np.random.choice(len(out))
lis = out[rand_i]

fig, axs = plt.subplots(len(lis))

for ax, li in zip(axs, lis, strict=True):
    ax.imshow(li)

plt.show()