import asyncio
import aiohttp
from pathlib import Path
from asyncio import Queue
import unicodedata
import pickle

def strip_diacritics(text):
    return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')

from alphabet import A

base_url = "https://scholarlyeditions.brill.com/library/passage/urn:cts:ancJewLit:hebBible.genesis.dsbo-leningrad:{}/text/"
data_dir = Path(__file__).parent / "data" / "bible"

async def download_chapter(session, queue):
    while not queue.empty():
        number = await queue.get()
        url = base_url.format(number)
        try:
            async with session.get(url) as response:
                response.raise_for_status()
                text = await response.text()
                file_path = data_dir / f"chapter_{number}.txt"
                file_path.write_text(text, encoding='utf-8')
                print(f"Saved chapter {number} to {file_path}")
        except Exception as e:
            print(f"Failed to download chapter {number}: {e}")
        finally:
            queue.task_done()

async def async_download():
    queue = Queue()
    for number in range(1, 51):
        queue.put_nowait(number)

    async with aiohttp.ClientSession() as session:
        workers = [asyncio.create_task(download_chapter(session, queue)) for _ in range(10)]
        await queue.join()
        for worker in workers:
            worker.cancel()

def download():
    data_dir.mkdir(parents=True, exist_ok=True)
    asyncio.run(async_download())

hebrew_to_enum = {
    'א': A.Alef,
    'ע': A.Ayin,
    'ב': A.Bet,
    'ד': A.Dalet,
    'ג': A.Gimel,
    'ה': A.He,
    'ח': A.Het,
    'כ': A.Kaf,
    'ך': A.Kaf_final,
    'ל': A.Lamed,
    'ם': A.Mem,
    'מ': A.Mem_medial,
    'ן': A.Nun_final,
    'נ': A.Nun_medial,
    'פ': A.Pe,
    'ף': A.Pe_final,
    'ק': A.Qof,
    'ר': A.Resh,
    'ס': A.Samekh,
    'ש': A.Shin,
    'ת': A.Taw,
    'ט': A.Tet,
    'ץ': A.Tsadi_final,
    'צ': A.Tsadi_medial,
    'ו': A.Waw,
    'י': A.Yod,
    'ז': A.Zayin
}

EXTRA_FILTER = {
    '׃', '־', '׀', '[', ']', '\n', '\xa0', '\u202a', '\u202c'
}


def encode():

    n_non_encoded = 0
    n_encoded = 0

    encoded = []


    with open(data_dir / "chapter_1.txt", "r", encoding="utf-8") as f:

        for line in f:
            if "Chapter" in line:
                continue

            if len(line) < 3:
                continue

            for word in line.split(" "):

                word = strip_diacritics(word)
                for char in word:
                    if char in EXTRA_FILTER or char.isspace():
                        continue
                    if char not in hebrew_to_enum:
                        non_encoded.add(char)
                        n_non_encoded += 1
                    else:
                        n_encoded += 1
                        encoded.append(hebrew_to_enum[char])

                encoded.append(A.Space)

    print(encoded)

    print(n_encoded, n_non_encoded)
    print(non_encoded)



if __name__ == "__main__":

    # download()

    non_encoded = set()

