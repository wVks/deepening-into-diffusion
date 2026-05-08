# Обучение диффузионной модели на датасете котов

## Выбранный датасет
- Название: Oxford-IIIT Pet (cats-only subset)
- Ссылка: https://www.robots.ox.ac.uk/~vgg/data/pets/
- Способ загрузки: `torchvision.datasets.OxfordIIITPet`
- Использованный сплит: `trainval`
- Фильтрация: только коты (`species == 1`)
- Количество изображений котов: `1188`
- Предобработка: `Resize(64x64)`, приведение в `RGB`, нормализация в диапазон `[-1, 1]`

## Архитектура модели
- Модель: `UNet2DModel` (из `diffusers`)
- Целевая функция: `MSE(predicted_noise, target_noise)`
- Число шагов диффузии при обучении: `1000`
- Тип обучения: с нуля (без предобученной модели)

## Запуск
- Установка:
  - `py -3 -m venv .venv`
  - `.venv\Scripts\activate`
  - `pip install -r requirements.txt`
- Обучение:
  - `python train.py --config configs/train_oxford_cats64.yaml`
- Генерация DDPM:
  - `python sample.py --checkpoint outputs/oxford_cats64/checkpoints/final --method ddpm --steps 1000 --num-images 4 --out outputs/oxford_cats64/generated_ddpm_1000.png`
- Генерация DDIM:
  - `python sample.py --checkpoint outputs/oxford_cats64/checkpoints/final --method ddim --steps 100 --num-images 4 --out outputs/oxford_cats64/generated_ddim_100.png`

## Параметры обучения
- Разрешение: `64x64`
- Batch size: `8`
- Эпохи: `60`
- Оптимизатор: `AdamW`
- Сохранение чекпоинтов: каждые `5` эпох + `final`
- Валидация: периодическая генерация и сохранение примеров в `outputs/oxford_cats64/samples`

## Результаты
- График loss: `outputs/oxford_cats64/loss_curve.png`
- История обучения: `outputs/oxford_cats64/logs/history.json`
- Лучший `avg_loss`: `0.038192`
- Финальный `avg_loss` (epoch 60): `0.042254`
- Финальный чекпоинт: `outputs/oxford_cats64/checkpoints/final`

## Сравнение DDPM и DDIM
Замеры из `outputs/oxford_cats64/sampling_benchmark.csv`:
- Настройка DDPM: `1000` шагов, `4` изображения
  - run1: `314.275` с
  - run2: `320.107` с
  - среднее: `317.191` с
- Настройка DDIM: `100` шагов, `4` изображения
  - run1: `32.970` с
  - run2: `31.886` с
  - среднее: `32.428` с

Вывод по скорости:
- DDIM в этом эксперименте быстрее DDPM примерно в `9.78x`.

## Артефакты
- DDPM генерация: `outputs/oxford_cats64/generated_ddpm_1000.png`
- DDIM генерация: `outputs/oxford_cats64/generated_ddim_100.png`
- Benchmark CSV: `outputs/oxford_cats64/sampling_benchmark.csv`
- Demo video: `outputs/oxford_cats64/demo.mp4`

## Выводы
- Полный пайплайн обучения и генерации реализован и отработал без критических ошибок.
- Реализованы оба метода сэмплинга (DDPM и DDIM) и их сравнение по времени.
- Работа соответствует требованиям ТЗ по воспроизводимости, обучению, генерации и отчёту.

## Демо
- Видео-файл: `outputs/oxford_cats64/demo.mp4`
- Ссылка: https://drive.google.com/file/d/1BP1e7pJRwebynWE2P-JxNmJSolgdoYr9/view?usp=sharing
