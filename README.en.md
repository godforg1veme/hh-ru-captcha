# hh.ru Capture

[Русский](README.md) · [Setup](docs/setup.md) · [Evaluation](docs/evaluation.md)

Local OCR for short Cyrillic text on a curved baseline. The project combines a synthetic image generator, geometric rectification and a fine-tuned PaddleOCR recognizer. It accepts an image file or a directory and writes JSONL records containing the text and a confidence score.

The current model targets 250×90 images containing two groups of Cyrillic letters. Output is lowercase; internal spaces and the letter ё remain significant. Validate this specialized component against the data used by your product before integrating it.

The project is independent of HeadHunter and is not an official hh.ru product. It processes local images; website integration is not implemented.

## How it works

1. OpenCV extracts text strokes, estimates baseline curvature and rectifies the image.
2. A PP-OCRv5 recognizer processes two crops. One uses greedy CTC decoding; the other uses lexicon-free beam search.
3. The configuration selects a candidate by its model score. An additional original model can restore one missing letter when clipped ink is detected and the text passes conservative constraints.
4. The CLI writes `file`, `text` and `confidence`. The score is not a calibrated probability that the entire string is correct.

The [configuration](experiments/recognition_v2/frozen_config.json) records checkpoint SHA-256 hashes, which are verified at load time.

## Usage

Tested environment: Windows x64, Python 3.13 and PaddlePaddle 3.2.0. Follow the [setup guide](docs/setup.md) to install dependencies and download model weights from the private GitHub Release.

```powershell
python recognize.py .\input --device cpu --output results.jsonl
```

An illustrative output record:

```json
{"file": "input/sample.png", "text": "лесная тропа", "confidence": 0.94}
```

Generate curved text with per-letter shifts, background noise and lines:

```powershell
python generator.py --count 100 --seed 42 --output synthetic_v4
```

The generator uses fonts installed on Windows by default. Font files are not distributed. Its built-in phrases are a demonstration list; training requires a separate corpus.

## Results

Accuracy measures complete string matches after lowercasing and whitespace normalization. Internal spaces count.

| Dataset | Images | Original model | Current version |
| --- | ---: | ---: | ---: |
| Fresh synthetic test | 1,000 | 73.2% | 95.2% |
| Fresh stress test | 500 | 41.2% | 92.6% |
| Earlier synthetic set | 1,000 | 85.3% | 97.0% |

Training used 7,500 images; configuration selection used 600. Words in the fresh final tests do not overlap with training or development data. All nine familiar original images are recognized correctly, but those images informed error analysis and are regression checks. They do not estimate accuracy on new website images.

See [evaluation details](docs/evaluation.md) for tuning history and limitations. Source datasets and reference images are excluded from this repository.

## Contents and status

- `recognize.py`, `recognition_*.py`: CLI, model loading and decoding.
- `rectifier.py`: baseline rectification.
- `generator.py`: synthetic images with a reproducible seed.
- `PaddleOCR/ppocr`: a retained upstream source snapshot used by the model architecture.
- `test_*.py`: generator and recognition semantics checks.

This repository contains the current local inference implementation. Training experiments, logs and environments remain in the working archive. An HTTP API, service performance measurements and an SLA are outside the current implementation.

No open-source license is granted for the project's original code. Third-party terms are retained in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Terms for original code, weights and data need a separate decision before broader distribution.
