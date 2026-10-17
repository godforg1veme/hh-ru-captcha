# Проверка качества / Evaluation

The reported version uses the frozen configuration in `experiments/recognition_v2/frozen_config.json`. Machine-readable counts and error rates are in [metrics.json](metrics.json). The source images, labels and training artifacts remain private, so this repository alone is insufficient to reproduce the full training experiment.

## Данные и выбор настроек / Data and selection

Training used 6,000 vocabulary-based images and 1,500 procedural stress images. Development used 400 and 200 images respectively. Vocabulary came from wordfreq 3.1.1; stress strings use generated Cyrillic syllables with additional repeated letters, ё and ъ. Actual word sets were checked for overlap across eight training, development and test splits; all pairwise intersections were empty.

A first test round exposed a known-original regression. It was then treated as regression data. An intermediate run was produced but its results were not inspected or used for selection. A fresh final test of 1,000 vocabulary-based and 500 stress images was generated with seed 2026101703; final settings were fixed before inspecting its answers.

The insertion safeguard was designed from a familiar original image. It had zero eligible cases on the 600-image development set. Its inclusion does not establish an independently measured development improvement.

## Результаты / Results

| Set | Before | After | Character error rate after |
| --- | ---: | ---: | ---: |
| Fresh synthetic, 1,000 | 732/1,000 | 952/1,000 | 0.435% |
| Fresh stress, 500 | 206/500 | 463/500 | 0.768% |
| Earlier synthetic, 1,000 | 853/1,000 | 970/1,000 | 0.342% |
| Known originals, 9 | 8/9 | 9/9 | 0% |

The baseline uses the original recognition weights with rectification. The comparable dynamic baseline scored 85.3% on the earlier synthetic set; a previous static inference run scored 85.7%. These are different execution paths and should not be substituted for one another.

Outputs are lowercased and whitespace is normalized; internal spaces, ё and Cyrillic versus Latin characters remain significant. The fresh synthetic result has a Wilson 95% interval of approximately 93.69% to 96.36%; the fresh stress interval is approximately 89.97% to 94.58%. These intervals describe the sampled synthetic distributions.

All 2,509 evaluation records were matched against their source labels and the metrics were recomputed. No rectification failures were recorded. CPU and GPU predictions matched on the nine originals. Recognition scores are not calibrated probabilities.

## Ограничения / Limits

Синтетические изображения близки по стилю к использованным образцам, но не покрывают все варианты исходных изображений. Девять оригиналов размечены визуально ассистентом и участвовали в разборе ошибок. Для вывода о качестве на новых изображениях нужен отдельный, независимо размеченный набор. Другой алфавит, значимый регистр, иной размер изображения и другая геометрия требуют собственной проверки.

Synthetic images match the studied style but do not cover every source variation. The nine originals were visually transcribed by an assistant and informed error analysis. Accuracy on new source images requires a separate, independently labeled test set. Other alphabets, case-sensitive output, image sizes and geometry require their own validation.
