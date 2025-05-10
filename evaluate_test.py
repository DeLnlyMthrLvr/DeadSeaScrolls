import textdistance
import os
from pathlib import Path

def evaluate(file_path_actual: str, hat: list[list[str]]):
    lines_actual = []
    with open(file_path_actual, "r", encoding="utf-8") as f:
        for line in f:
            lines_actual.append(line.strip().replace(" ", ""))

    for idx in range(len(lines_actual)):
        levenshtein_normalized = textdistance.levenshtein.normalized_distance(lines_actual[idx], hat[idx][0])
        print(f"file: {os.path.basename(file_path_actual)}, line: {idx}: \n "
              f"actual:    {lines_actual[idx]} \n"
              f"predicted: {hat[idx][0]} \n"
              f"levenshtein_normalized: {levenshtein_normalized}")
    


def test_evaluate():
    actual_file = os.path.join(Path.cwd(), "data", "sample-test-2025", "Lines", "25-Fg001_characters.txt")
    hat = [
    ["ב֯ו֯דכהו֯"],
    ["‎ע֯יואמןא"],
    ["ב֯כ̇ו̇ר֯י֯ם֯ זככרה̇ א֯נכי֯ מ̇כע֯ד֯"],
    ["ו֯נדבות ר֯צוכנכה אשר֯ צוכיתה"],
    ["גי֯ש לפניכית מעסי֯"],
    ["לבם עלירץ כה̇יו̇ת̇ ק"],
    ["‎כ̇הכיביו֯ם֯ ה"],
    ["א̇ר֯הקכדשת"],
    ["א֯ת֯שגר"]
    ]
    evaluate(actual_file, hat)

if __name__ == "__main__":
    test_evaluate()