# **Многокамерный трекинг людей в реальном времени с использованием нейросетевых методов**

![Пример работы системы](assets/demo.gif)

## **Описание проекта**
Программное обеспечение для **обнаружения и трекинга людей** в видеопотоке с использованием нейросетевых моделей.
Основные возможности:
- Детекция людей с помощью **YOLOv11n**.
- Трекинг на основе **ReID (OSNet-x1.0)**.
- Поддержка работы с **несколькими камерами**.
- Оценка качества с помощью **MOTmetrics**.

## **Технологии**
- **Язык**: Python 3.10
- **Библиотеки**:
  - PyTorch, Ultralytics (YOLO), TorchReid (ReID)
  - OpenCV (обработка видео), Scipy (оптимизация), Accelerate (ускорение обучения)
- **Модели**:
  - Детектор: **YOLOv11n**
  - Feature extractor: **OSNet-x1.0**

## **Установка и настройка**

### **1. Подготовка окружения**
```bash
# Установка conda (если не установлен)

# Клонирование репозиториев
git clone https://github.com/KaiyangZhou/deep-person-reid.git
git clone https://github.com/Qolorerr/People-tracking.git

cd deep-person-reid/

# Создание виртуального окружения
conda create --name tracker python=3.10
conda activate tracker

# Установка зависимостей для torchreid
pip install -r requirements.txt

# Установка PyTorch (под вашу видеокарту)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118# для CUDA 11.8

# Установка torchreid
python setup.py develop

# Переход в папку проекта
cd ../People-tracking

# Установка зависимостей
pip install -r requirements.txt
conda install pyqt~=5.15.10

```

### **2. Настройка конфигурации**
- Указать пути к датасетам в `config/config.yaml`.
- Указать IP-адреса камер (если используется live-режим).

## **Запуск**

### **1. Подготовка датасета (опционально)**
```bash
python crop_sportsmot.py# обрезка SportsMOT для ускорения обучения
```

### **2. Обучение модели**
```bash
python train.py
```

### **3. Тестирование на датасете**
```bash
python test.py
```

### **4. Запуск в реальном времени**
```bash
python live_test.py
```

## **Демонстрация работы**
![Пример работы системы](assets/demo.gif)

## **Тестирование**
- Для оценки качества трекинга используется `test.py` (на датасете).
- Для live-тестирования — `live_test.py`.

## **Планы по развитию**
- Улучшение скорости обработки видео.
- Повышение точности трекинга.
- Добавление поддержки большего числа камер.

## **Контакты**
- **Автор**: Иванов Михаил 
- **Email**: [qolorer@gmail.com](mailto:qolorer@gmail.com)
- **GitHub**: [Qolorerr](https://github.com/Qolorerr)
