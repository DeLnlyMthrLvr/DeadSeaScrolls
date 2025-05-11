from pathlib import Path
import sys
import numpy as np
from PIL import Image

from linesegmentation.unet import LineSegmenter
from segmentation.unet import CharSegmenter
from alphabet import token_to_char

# Best unet checkpoints
LINE_UNET_BEST = "20250507_203729_wide_unet_largek_fixed_final"
CHAR_UNET_BEST = "20250510_152746_CCE_final"


def load_line_segmenter(run_name: str = LINE_UNET_BEST) -> LineSegmenter:
    runs_folder = Path(__file__).parent / "linesegmentation" / "runs"
    model = LineSegmenter.load(runs_folder / run_name)
    return model

def load_char_segmenter(run_name: str = CHAR_UNET_BEST) -> CharSegmenter:
    runs_folder = Path(__file__).parent / "segmentation" / "runs"
    model = CharSegmenter.load(runs_folder / run_name)
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

        with open(folder / f"{name}_characters.txt", "w", encoding="utf-8") as f:
            f.write(transcription)

    print(f"Results located in {str(folder.relative_to(folder.parent.parent))}")


def tokens_to_transcriptions(token_batches: list[list[list[int]]]) -> list[str]:

    documents = []
    for token_lines in token_batches:

        char_lines = []

        for tokens in token_lines:
            char_line = ''.join([token_to_char[token] for token in tokens])
            char_lines.append(char_line)

        document = '\n'.join(char_lines)
        documents.append(document)

    return documents

def pipeline(folder: Path):

    print("Loading stuff to RAM")
    images, names = load_images_names(folder)
    line_segmenter = load_line_segmenter()
    char_segmenter = load_char_segmenter()

    print("Segmenting lines")
    all_line_images = line_segmenter.process_heterogenous_images(images)

    print("Segmenting characters")
    tokens = char_segmenter.process_heterogenous_images(all_line_images)
    transcriptions = tokens_to_transcriptions(tokens)

    write_results(names, transcriptions)


def test():
    """Our internal evaluation
    """

    import Levenshtein
    from pathlib import Path

    hat_f = Path(__file__).parent / "results"
    truth_f = Path(__file__).parent / "data" / "test_answers"

    def test_file(name: str):

        with open(hat_f / name, "r", encoding="utf-8") as f:
            hat = f.read()

        with open(truth_f / name, "r", encoding="utf-8") as f:
            truth = f.read()

        print(name, Levenshtein.ratio(truth, hat))

    for ent in truth_f.iterdir():
        test_file(ent.name)


if __name__ == "__main__":

    print(sys.argv[1])
    path = Path(sys.argv[1])
    pipeline(path)
    # test()