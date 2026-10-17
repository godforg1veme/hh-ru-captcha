# Установка / Setup

Проверенная платформа / Tested platform: Windows x64, Python 3.13, CPU inference. Execute these PowerShell commands from the repository root.

## Окружение / Environment

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`PaddleOCR/ppocr` is a source snapshot retained from the working project. Its Apache 2.0 license is included. This is a subset of upstream, not a complete PaddleOCR distribution. The upstream commit was not recorded in the original downloaded archive; the Git commit of this repository identifies the retained snapshot.

## Веса / Weights

Установите GitHub CLI и авторизуйтесь в аккаунте с доступом к репозиторию. / Install GitHub CLI and authenticate with an account that has repository access.

```powershell
gh auth login
New-Item -ItemType Directory -Force experiments/recognition_v2/balanced_training | Out-Null
gh release download v0.1.0 --repo godforg1veme/hh-ru-capture --pattern best.pdparams --dir experiments/recognition_v2/balanced_training
gh release download v0.1.0 --repo godforg1veme/hh-ru-capture --pattern pretrained_dynamic.pdparams --dir experiments/recognition_v2
```

При повторном запуске GitHub CLI откажется перезаписывать существующие файлы. Проверьте их перед заменой. / GitHub CLI refuses to overwrite existing downloads. Inspect existing files before replacing them.

| Asset | SHA-256 |
| --- | --- |
| `best.pdparams` | `d1771fbc6a648636b0541225f9f7a92813e70c7f352496f773938ffa88472598` |
| `pretrained_dynamic.pdparams` | `3d25b99bef7ba60980b19136210e866da04fed835cbc9080a10c818ae1e477b2` |

The configuration verifies both hashes when the recognizer starts. The character dictionary and architecture YAML are tracked in Git. The original upstream pretrained file is not required for normal CLI inference with these two assets.

## Запуск и проверки / Run and checks

```powershell
python recognize.py .\input --device cpu --output results.jsonl
python -m unittest test_recognition -v
python -m unittest test_generator -v
```

The recognition suite does not need checkpoint files. Its dataset overlap audit is skipped when the private experiment datasets are absent; nine recognition semantics tests remain. The five generator tests require the Windows fonts used by the generator.

Для GPU нужен совместимый пакет `paddlepaddle-gpu` вместо `paddlepaddle` и подходящие CUDA/cuDNN-библиотеки. Текущие веса проверены на RTX 3050 с PaddlePaddle 3.2.0, CUDA 12.6 и cuDNN 9. Это не универсальный установщик GPU-окружения. / GPU inference requires `paddlepaddle-gpu` in place of the CPU package and matching CUDA/cuDNN libraries. The weights were tested on an RTX 3050 with PaddlePaddle 3.2.0, CUDA 12.6 and cuDNN 9; this guide does not provide a general GPU installer.

```powershell
python recognize.py .\input --device gpu:0 --output results.jsonl
```

Без весов можно отдельно использовать генератор и выпрямление. / Generation and rectification can run without model weights.

```powershell
python generator.py --count 100 --seed 42 --output synthetic_v4
python rectifier.py .\input --output rectified
```
