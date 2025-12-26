# Отчет по HW1: Training Loop и Linear Probe на ViT-Tiny

## Цель работы
Реализовать полный пайплайн обучения: подготовка данных, sanity-checks, обучение
базовой CNN и линейного пробинга ViT-Tiny, профилирование и оценка качества на
валидации.

## Данные
- Датасет: Oxford-IIIT Pet (`timm/oxford-iiit-pet`), 37 классов.
- Формат: `data/oxford_pets/{train,val}/{class_name}/*.jpg`.
- Статистика по sanity-checks:
  - train: 3680 изображений, min/max на класс: 93/100.
  - val: 3669 изображений, min/max на класс: 88/100.
- Аугментации и нормализация:
  - train: `RandomResizedCrop`, `RandomHorizontalFlip`, `Normalize`.
  - val: `Resize`, `CenterCrop`, `Normalize`.
  - Используются ImageNet mean/std.

## Модели и обучение
### CNN baseline
- Архитектура: 3 сверточных блока (64/128/256) + BN/ReLU/MaxPool, затем
  `AdaptiveAvgPool` и `Linear` голова с `Dropout=0.2`.
- Обучение по умолчанию: AdamW, `lr=1e-3`, `batch_size=32`, `epochs=50`,
  косинусный scheduler.
- Добавлена проверка на переобучение на 1 батче.

### ViT-Tiny linear probe
- Модель: `vit_tiny_patch16_224` из `timm`, предобученная.
- Бэкбон заморожен, обучается только классификационная голова.
- Обучение по умолчанию: Adam, `lr=1e-3`, `batch_size=32`, `epochs=10`.

## Воспроизводимость и логи
- Фиксация seed и детерминизм включены.
- Логи TensorBoard пишутся в `runs/`.

## Профилирование
- Использован `torch.profiler` (50–100 шагов).
- Трейсы: `artifacts/profiler/cnn`, `artifacts/profiler/vit`.

## Оценка качества
Метрики считаются на валидации: accuracy и macro-F1, сохраняются матрицы
ошибок в PNG/CSV.

Последние значения из `artifacts/metrics.csv`:

| model | split | accuracy | macro_f1 |
| --- | --- | --- | --- |
| cnn | val | 0.230581 | 0.218526 |
| vit | val | 0.791224 | 0.788831 |

Матрицы ошибок:
- `artifacts/confusion_cnn.png`
- `artifacts/confusion_vit.png`

## Артефакты
- Sanity-checks: `artifacts/data_sanity/` (таблицы классов и гриды).
- Чекпойнты и конфиги:
  - `artifacts/cnn/` (`checkpoint_best.pt`, `config.json`, `metrics.json`).
  - `artifacts/vit/` (`checkpoint_best.pt`, `config.json`, `metrics.json`).
- Метрики и матрицы: `artifacts/metrics.csv`, `artifacts/confusion_*.{png,csv}`.

## Команды для воспроизведения
```bash
python -m src.prepare_oxford_pets --output-dir data/oxford_pets
python -m src.sanity_data --data-dir data/oxford_pets
python -m src.train_cnn --data-dir data/oxford_pets --run-name cnn_full
python -m src.train_cnn --data-dir data/oxford_pets --overfit --overfit-only \
  --overfit-log --overfit-batches 1
python -m src.train_vit --data-dir data/oxford_pets --run-name vit_full
python -m src.train_cnn --data-dir data/oxford_pets --profile --profile-only
python -m src.train_vit --data-dir data/oxford_pets --profile --profile-only
python -m src.eval --model cnn --checkpoint artifacts/cnn/checkpoint_best.pt \
  --data-dir data/oxford_pets
python -m src.eval --model vit --checkpoint artifacts/vit/checkpoint_best.pt \
  --data-dir data/oxford_pets
```

## Краткие выводы
- ViT-Tiny с линейным пробингом существенно превосходит CNN по accuracy и
  macro-F1 на валидации.
- CNN, обученная с нуля на 37 классах, показывает слабую обобщающую способность
  при текущей емкости модели и настройках обучения.
