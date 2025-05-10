


# Custom TrOCR (Not used in actual inference)

A custom made TrOCR model. This model underperformed and is for that reason not used.

---

## Scripts

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