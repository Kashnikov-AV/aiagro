# models/ — веса детектора культур (GOG-режим)

Папка для обученных YOLO-моделей. Этап `YoloCropStage` (core/pipeline.py)
ищет веса в приоритете: crop_yolo.pt -> crop_yolo.onnx -> best.pt -> yolov8n.pt.

1. `crop_yolo.pt`   — ваша кастомная модель (YOLOv8-seg/detect), дообученная
                      на культуры проекта. Рекомендуемый вариант.
2. `crop_yolo.onnx` — та же модель в ONNX (`yolo export model=best.pt format=onnx`)
                      — быстрее на CPU/Raspberry Pi.
3. `best.pt`        — стандартное имя весов после `yolo train` в Ultralytics.
4. `yolov8n.pt`     — предобученная COCO-модель (класс "plant"), скачана как
                      базовый вариант для отладки конвейера без своей разметки.

## Как получить свою модель (Ultralytics)

    pip install ultralytics
    # разметка в формате YOLO (Roboflow / CVAT / LabelImg), data.yaml с классом crop
    yolo segment train model=yolov8n-seg.pt data=data.yaml epochs=100 imgsz=640
    cp runs/segment/train*/weights/best.pt models/crop_yolo.pt

Сегментная модель (-seg) предпочтительнее detekционной: её маска точнее
повторяет форму растения, и обратная маска культур не срезает соседние сорняки.
Для det-модели маска строится эллипсом по bounding box.

Файлы *.pt/*.onnx в git не коммитятся (см. .gitignore).
