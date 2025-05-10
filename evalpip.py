
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