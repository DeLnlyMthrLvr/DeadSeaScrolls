


# Oh dear U-NET. Dead sea scrolls character identification

This repository implements character identification pipeline:
1. Binarized scroll images
2. U-NET line segmentation
3. Flood-fill algorithm for unique line identification
4. U-NET character segmentation on the line images
5. Center identification algorithm for extracting unique characters and their x-coordinate
6. Hebrew transcriptions (no spaces included)

## Quickly explained

A brief summary of every component in the pipeline (feel free to skip this section if you are not interested in that).

As a preprocessing step we have cropped the alphabet images (used for synthetic data generation) using a connected component algorithm.

### `U-NET line segmentation`

We have took the clasiscal idea of line identification by projecting the binarized mask on the y axis (mask.sum(axis=1)) and finding the peaks and generalized it for bended lines in the U-NET.

Our customed made U-NET encoder reduces gradually only the x-axis (similar to mask.sum(axis=1)) and we have also implemented a custom 1D-horizontal-PixelShuffle layer for the decoder which increases the width resolution only (restoring back to the original image size). This allows the U-NET to work on a slightly projected spaces (making it very fast and easy to train) whilst allowing for bended lines.

### `Flood-fill algorithm for unique line identification`

Since the U-NET can produce lines that touch with each other our special flood-fill algorithm uniquely identifies those lines by launching N simultaneous DFS flood-fill algorithms. Each DFS "progress" is guided by a PriorityQueue which priotizes DFS that are the most "behind" (going from right to left of the image). Conseqeuntly even intersecting lines will be uniquely identified.

### `U-NET character segmentation on the line images`

Classical unet with 27 output channels for each character. Trained on our synthetic images which include 3 types of noise

1. Cutout noise to simulate the gaps found in the scrolls
2. Warp noise which simulates a slight character/line bending
3. Multiple octave-Perlin noise which simulates high-frequency wear damage

### `Center identification algorithm`

To identify unique characters from the segmentation masks and their order our center identifaction algorithm is used which uses a noise-resilliant connected components algorithm.

## Installation and running

1. `git clone` our repository
2. (Optional) initialize and activate a virtual environment
3. `pip install -r requirements.txt`
4. Install your version of pytorch (since everybody uses a different one)
5. To run our pipeline simply execute the following command ``python pipeline.py ./data/test_jpg/``
6. Our pipeline accepts `*.jpg` images only, does not work with color images (the jpg is casted to single channels) and the results will be located in `/project_folder/results` as `*.txt` files with the same base name (plus `_characters.txt`) as the images. Our output does not include spaces (but it includes newlines).


## File overview

We have tried numerous ideas, thus, the poject will include many redundant files. The most important ones are described in this readme. The ones included in the readme are somewhat clean and commented (the others, not important are messy).

### `./`

- `alphabet.py` enum for hebrew characters, code for loading ngrams and the alphabet images
- `bible.py` we have also downloaded a hebrew bible passages which we are using for synthetic data generation together with the ngrams
- `crop_alphabet.py` creates a cleaner cropped alphabet images
- `noise_designer.py` code for visualizing various noise levels and generating our training datasets
- `noise.py` Perlin noise manager
- `pipeline.py` For running the end-to-end pipeline
- `synthetic.py` Synthetic scrolls creation code, includes margins, cutout noise (big gaps), warp noise, line-spacing, character spacing, line-segmentation masks, character segmentation masks.


### `./linesegmentation`

- `floodfill.py` the already explained floodfill algorithm for unique line identification (from unet)
- `train.py` trainign the line segmentation unet
- `unet.py` line segmentation unet architecture
- `/runs` where the training runs will appear (the best one is included already)

### `./segmentation`

Character segmentation

- `centers.py` for unique character and their center identification (from unet)
- `train.py` trainign the character segmentation unet
- `unet.py` character segmentation unet architecture
- `/runs` where the training runs will appear (the best one is included already)

## Custom TrOCR (Not used in actual inference)

A custom made TrOCR model. This model underperformed and is for that reason not used.

---

### Scripts

### `ocr/data_loader.py`
- Defines dataset classes:
  - Bible text
  - N-grams
  - Bible with noise
  - N-grams with noise
  - Mixed datasets
- Uses `ocr/image_creator.py` to render text as images.

### `ocr/image_creator.py`
- Renders text into image format.
- Supports padding, grayscale conversion, and font selection.

### `ocr/tokenizer.py`
- `Tokenizer` class for:
  - Encoding text to token IDs
  - Decoding token IDs to text
  - Vocabulary management

### `ocr/ocr_model.py`
- Custom TrOCR-based model.
- Uses a ViT encoder and an autoregressive text decoder.

### `ocr/train.py`
- Trains the OCR model on synthetic data.
- Handles model saving, loss logging, and evaluation.

### `ocr/inference.py`
- Loads a trained model and runs inference on new images.
- Outputs predicted text.

---