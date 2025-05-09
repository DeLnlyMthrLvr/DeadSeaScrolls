from pathlib import Path
import sys
import numpy as np
from PIL import Image

from linesegmentation.unet import LineSegmenter

UNET_BEST_PROJECTED = "20250507_203729_wide_unet_largek_fixed_final"

def load_line_segmenter(run_name: str = UNET_BEST_PROJECTED) -> LineSegmenter:
    runs_folder = Path(__file__).parent / "linesegmentation" / "runs"
    model = LineSegmenter.load(runs_folder / run_name)
    return model


def load_images_names(folder: Path) -> tuple[list[np.ndarray], list[str]]:

    images = []
    names = []
    for ent in folder.iterdir():

        if not ent.is_file():
            continue

        if not ent.name.endswith(".jpg"):
            continue

        img = Image.open(ent).convert("L")
        img_array = np.array(img)
        images.append(img_array)

        names.append(ent.stem)

    return images, names


def delete_files_in_folder(path: Path):
    for ent in path.iterdir():
        if ent.is_file():
            ent.unlink()


def write_results(names: list[str], transcriptions: list[str]):

    folder = Path(__file__).parent / "results"
    folder.mkdir(exist_ok=True)

    # Cleanup from last time
    delete_files_in_folder(folder)

    for name, transcription in zip(names, transcriptions, strict=True):

        with open(folder / f"{name}_characters.txt", "w") as f:
            f.write(transcription)

    print(f"Results located in {str(folder.relative_to(folder.parent.parent))}")


def dummy_transformer(all_line_images: list[list[np.ndarray]]) -> list[str]:
    from bible import hebrew_to_enum
    from alphabet import MEAN_CHAR_WIDTH

    transcriptions = []
    characters = list(hebrew_to_enum.keys())

    for line_images in all_line_images:

        transcription = []
        for line_image in line_images:

            h, w = line_image.shape
            n = int(round(w / MEAN_CHAR_WIDTH))
            chars = np.random.choice(characters, size=n, replace=True)

            transcription.append(chars)

        transcriptions.append('\n'.join(transcription))

    return transcriptions


def pipeline(folder: Path):

    print("Loading stuff to RAM")
    images, names = load_images_names(folder)
    line_segmenter = load_line_segmenter()

    print("Segmenting lines")
    all_line_images = line_segmenter.process_heterogenous_images(images)

    print("OCR")
    transcriptions = dummy_transformer(all_line_images)

    write_results(names, transcriptions)


if __name__ == "__main__":

    print(sys.argv[1])
    path = Path(sys.argv[1])
    pipeline(path)