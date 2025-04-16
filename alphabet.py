from pathlib import Path
from enum import Enum
from PIL import Image
import numpy as np


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
    Mem_medial = "Mem_medial"
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
    A.Zayin: 26
}

def alphabet_path():
    return Path(__file__).parent.parent / "data" / "alphabet"

def load_alphabet() -> dict[str, list[np.ndarray]]:

    results = dict()

    for ent in alphabet_path().iterdir():
        if not ent.is_dir():
            continue

        name = ent.name
        images = []

        for ent_image in ent.iterdir():

            if not ent_image.is_file():
                continue

            if not ent_image.name.endswith(".pgm"):
                continue

            img = Image.open(ent_image)
            img_array = np.array(img)

            images.append(img_array)

        results[name] = images

    return results


if __name__ == "__main__":
    results = load_alphabet()

    print(results[A.Alef.value][0])