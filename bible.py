import asyncio
import random
from typing import Literal
import aiohttp
from pathlib import Path
from asyncio import Queue
import unicodedata
import pickle

def strip_diacritics(text):
    return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')

from alphabet import A, hebrew_to_enum

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


EXTRA_FILTER = {
    '׃', '־', '׀', '[', ']', '\n', '\xa0', '\u202a', '\u202c'
}


def encode():

    non_encoded = set()
    n_non_encoded = 0
    n_encoded = 0

    encoded = []

    for i in range(1, 51):

        with open(data_dir / f"chapter_{i}.txt", "r", encoding="utf-8") as f:

            document = []

            for line in f:
                if "Chapter" in line:
                    continue

                if len(line) < 3:
                    continue

                for word in line.split(" "):

                    word = strip_diacritics(word)

                    add_word = []
                    for char in word:
                        if char in EXTRA_FILTER or char.isspace():
                            continue
                        if char not in hebrew_to_enum:
                            non_encoded.add(char)
                            n_non_encoded += 1
                        else:
                            n_encoded += 1
                            add_word.append(hebrew_to_enum[char])

                    document.append(add_word)

        encoded.append(document)

        print(n_encoded, n_non_encoded)
        print(non_encoded)

    with open(Path(__file__).parent / "data" / "bible.pickle", "wb") as f:
        pickle.dump(encoded, f)


Word = list[A]
Document = list[Word]

class BibleTexts:

    def __init__(self, max_sequence_length: int = 150):

        self.documents: list[Document] = None
        self.max_sequence_length = max_sequence_length
        self.flatten: Document = None

        self.load_documents()

    def load_documents(self):
        with open(Path(__file__).parent / "data" / "bible.pickle", "rb") as f:
            self.documents = pickle.load(f)

        self.flatten = []
        for doc in self.documents:
            self.flatten.extend(doc)


    def sample_passages(self, N: int) -> list[list[A]]:

        L = len(self.flatten)
        passages = []
        for _ in range(N):
            n_consumed_tokens = 0

            i_pointer = random.randint(0, L - 1)
            passage = []
            while n_consumed_tokens < self.max_sequence_length:

                if i_pointer >= L:
                    i_pointer = i_pointer - L

                word = self.flatten[i_pointer]

                if n_consumed_tokens > 0:
                    passage.append(A.Space)
                    n_consumed_tokens += 1

                passage.extend(word)

                n_consumed_tokens += len(word)
                i_pointer += 1

            passage = passage[:self.max_sequence_length]
            passages.append(passage)

        return passages


if __name__ == "__main__":

    # download()
    encode()
