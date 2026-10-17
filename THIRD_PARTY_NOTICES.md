# Third-party notices

## PaddleOCR and model lineage

The `PaddleOCR/ppocr` source snapshot and the selected architecture configuration come from [PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR). Copyright notices remain in the source files. The upstream Apache License 2.0 is retained at [PaddleOCR/LICENSE](PaddleOCR/LICENSE).

The recognizer was initialized from the upstream `eslav_PP-OCRv5_mobile_rec_pretrained` model. The selected checkpoint is a fine-tuned derivative; the rescue checkpoint retains the original model state in the training implementation's format. The release identifies both files by SHA-256. Inclusion in this private repository does not assert ownership of upstream models or grant rights beyond their applicable terms.

## Vocabulary

The private training corpus used wordfreq 3.1.1 by Robyn Speer: https://github.com/rspeer/wordfreq. Upstream code is Apache 2.0; upstream data includes CC BY-SA 4.0 terms and source-specific acknowledgements. The retained [license and acknowledgements](docs/third-party/wordfreq-LICENSE.txt) document that provenance. The word lists, labeled datasets and source images are not distributed here.

## Fonts and Python dependencies

The generator reads locally installed font files. Windows font files are not included, and their licenses remain with their respective owners. Python dependencies retain their own licenses; the requirements files are installation manifests, not a replacement for those terms.

## Original project code and name

No open-source license is granted for the original project code. Third-party license grants remain in effect for their own components. A license for original code or broader model distribution must be chosen separately by the owner.

The name `hh.ru Capture` does not imply endorsement or affiliation with HeadHunter. No HeadHunter logo or reference image is distributed in this repository.
