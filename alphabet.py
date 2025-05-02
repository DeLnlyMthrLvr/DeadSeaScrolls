from pathlib import Path
from enum import Enum
from PIL import Image
import numpy as np
import pandas as pd

MEAN_NGRAM_CHAR = 3.742667
MAX_NGRAM_CHAR = 10
MEAN_CHAR_WIDTH = 37.8
MEAN_CHAR_HEIGHT = 44.4

class A(Enum):
    Alef = "Alef"
    Ayin = "Ayin"
    Bet = "Bet"
    Dalet = "Dalet"
    Gimel = "Gimel"
    He = "He"
    Het = "Het"
    Kaf = "Kaf"
    Kaf_final = "Kaf-final"
    Lamed = "Lamed"
    Mem = "Mem"
    Mem_medial = "Mem-medial"
    Nun_final = "Nun-final"
    Nun_medial = "Nun-medial"
    Pe = "Pe"
    Pe_final = "Pe-final"
    Qof = "Qof"
    Resh = "Resh"
    Samekh = "Samekh"
    Shin = "Shin"
    Taw = "Taw"
    Tet = "Tet"
    Tsadi_final = "Tsadi-final"
    Tsadi_medial = "Tsadi-medial"
    Waw = "Waw"
    Yod = "Yod"
    Zayin = "Zayin"
    Space = " "


Ngrams = list[tuple[A, ...]]

str_to_enum = {
    e.value: e for e in A
}

char_token = {
    A.Alef: 0,
    A.Ayin: 1,
    A.Bet: 2,
    A.Dalet: 3,
    A.Gimel: 4,
    A.He: 5,
    A.Het: 6,
    A.Kaf: 7,
    A.Kaf_final: 8,
    A.Lamed: 9,
    A.Mem: 10,
    A.Mem_medial: 11,
    A.Nun_final: 12,
    A.Nun_medial: 13,
    A.Pe: 14,
    A.Pe_final: 15,
    A.Qof: 16,
    A.Resh: 17,
    A.Samekh: 18,
    A.Shin: 19,
    A.Taw: 20,
    A.Tet: 21,
    A.Tsadi_final: 22,
    A.Tsadi_medial: 23,
    A.Waw: 24,
    A.Yod: 25,
    A.Zayin: 26,
    A.Space: 27
}

def alphabet_path(cropped: bool = True):
    return Path(__file__).parent / "data" / ("alphabet_cropped" if cropped else "alphabet")

def load_alphabet(cropped: bool = True, include_paths: bool = False) -> dict[str, list[np.ndarray]]:

    results = dict()

    for ent in alphabet_path(cropped).iterdir():
        if not ent.is_dir():
            continue

        name = ent.name
        images = []

        for ent_image in ent.iterdir():

            if not ent_image.is_file():
                continue

            if not ent_image.name.endswith(".pgm"):
                continue

            img = Image.open(ent_image).convert("L")
            img_array = np.array(img)

            if include_paths:
                images.append((img_array, ent_image))
            else:
                images.append(img_array)

        results[name] = images

    return results

def load_n_grams() -> tuple[Ngrams, np.ndarray, np.ndarray]:
    df = pd.read_csv(alphabet_path().parent / "ngrams" / "ngrams_frequencies_withNames.csv")
    df["Names"] = df["Names"].str.replace("Tasdi-final", "Tsadi-final")
    df["Names"] = df["Names"].str.replace(r'Tsadi(?!-final)', "Tsadi-medial", regex=True)

    ngrams = []
    for names in df["Names"]:
        list_names = names.split("_")
        ngrams.append(tuple(str_to_enum[name] for name in list_names))

    frequencies = df["Frequencies"].to_numpy()
    ngram_tokens = np.arange(len(ngrams))

    return ngrams, frequencies, ngram_tokens


def sample_ngrams(N: int, ngrams: Ngrams = None, frequencies: np.ndarray = None, ngram_tokens: np.ndarray = None) -> tuple[Ngrams, np.ndarray]:

    if ngrams is None:
        ngrams, frequencies, ngram_tokens = load_n_grams()

    weights = frequencies / frequencies.sum()
    inds = np.random.choice(len(ngrams), size=N, replace=True, p=weights)

    sampled_ngrams = [ngrams[i] for i in inds]

    return sampled_ngrams, ngram_tokens[inds]



if __name__ == "__main__":
    results = load_alphabet()

    print(results[A.Alef.value][0])