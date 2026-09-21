import numpy as np
import cv2


class PlantDetector:
    """класс для обнаружения растений"""

    def __init__(self, index_type='exg', downscale_factor=0.5):
        self.index_type = index_type
        self.downscale_factor = downscale_factor

        # Параметры размеров растений
        self.small_area_factor = 0.0001
        self.medium_scale = 9
        self.large_scale = 4

        # ядро для морфологии
        self.kernel = np.ones((3, 3), np.uint8)

    def calculate_index(self, img):
        """вычисление цветового индекса с векторизацией"""
        # Работаем напрямую с массивом, без разделения каналов
        R = img[:, :, 2].astype(np.float32)
        G = img[:, :, 1].astype(np.float32)
        B = img[:, :, 0].astype(np.float32)

        sum_RGB = R + G + B + 1e-7

        # Векторизованные операции
        if self.index_type == 'exg':
            return 2 * G / sum_RGB - R / sum_RGB - B / sum_RGB

        elif self.index_type == 'exr':
            return 1.4 * R / sum_RGB - G / sum_RGB

        elif self.index_type == 'exgr':
            return 3 * G / sum_RGB - 2.4 * R / sum_RGB - B / sum_RGB

        elif self.index_type == 'cive':
            return 0.441 * R - 0.811 * G + 0.385 * B + 18.787


    def apply_otsu(self, index, manual_threshold=None):
        """Пороговая обработка. Если manual_threshold задан — используется он, иначе Оцу."""
        invert = self.index_type in ['exr', 'cive']

        index_min, index_max = index.min(), index.max()
        # защита от деления на ноль, если картинка однородная
        if index_max == index_min:
            return np.zeros_like(index, dtype=np.uint8)
        index_norm = ((index - index_min) * 255 / (index_max - index_min)).astype(np.uint8)

        # --- ВЫБОР ПОРОГА ---
        if manual_threshold is not None:
            threshold = int(manual_threshold)
        else:
            # стандартный расчёт Оцу
            hist = cv2.calcHist([index_norm], [0], None, [256], [0, 256]).flatten()
            pixel_count = index_norm.size
            mean_weight = np.cumsum(hist)
            mean_intensity = np.cumsum(hist * np.arange(256))

            max_var, threshold = 0, 128
            for t in range(1, 255):
                w0, w1 = mean_weight[t], pixel_count - mean_weight[t]
                if w0 == 0 or w1 == 0:
                    continue
                m0 = mean_intensity[t] / w0
                m1 = (mean_intensity[255] - mean_intensity[t]) / w1
                var = w0 * w1 * (m0 - m1) ** 2
                if var > max_var:
                    max_var, threshold = var, t

        binary = (index_norm > threshold).astype(np.uint8) * 255
        return cv2.bitwise_not(binary) if invert else binary

    def morph_processing(self, binary):
        """морфологическая обработка"""
        # Комбинированная операция для скорости
        bitmap = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self.kernel, iterations=1)
        bitmap = cv2.morphologyEx(bitmap, cv2.MORPH_OPEN, self.kernel, iterations=1)
        bitmap = cv2.dilate(bitmap, self.kernel, iterations=1)

        return bitmap

    def detect_plants(self, bitmap, original_shape):
        """обнаружение контуров с предварительной фильтрацией"""
        h_img, w_img = original_shape
        img_area = h_img * w_img

        small_s = img_area * self.small_area_factor
        medium_s = small_s * self.medium_scale
        large_s = medium_s * self.large_scale

        # Поиск контуров
        contours, _ = cv2.findContours(
            bitmap,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_TC89_L1
        )

        # Векторизованная фильтрация по площади
        areas = np.array([cv2.contourArea(cnt) for cnt in contours])

        small_mask = (areas > small_s) & (areas <= medium_s)
        medium_mask = (areas > medium_s) & (areas <= large_s)
        large_mask = areas > large_s

        small_contours = [contours[i] for i in np.where(small_mask)[0]]
        medium_contours = [contours[i] for i in np.where(medium_mask)[0]]
        large_contours = [contours[i] for i in np.where(large_mask)[0]]

        return small_contours, medium_contours, large_contours

    def process_frame(self, frame):
        """обработка кадра"""
        # 1. Уменьшение разрешения для ускорения (сохраняем оригинал для отображения)
        original_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.downscale_factor < 1.0:
            h, w = original_rgb.shape[:2]
            new_size = (int(w * self.downscale_factor), int(h * self.downscale_factor))
            frame_small = cv2.resize(original_rgb, new_size, interpolation=cv2.INTER_LINEAR)
        else:
            frame_small = original_rgb

        # 2. Быстрое вычисление индекса
        index = self.calculate_index(frame_small)

        # 3. Визуализация индекса (цветовая карта)
        index_viz = self.visualize_index(index)

        # 4. Оптимизированная пороговая обработка
        binary = self.apply_otsu(index)

        # 5. Морфологическая обработка
        bitmap = self.morph_processing(binary)

        # 6. Обнаружение контуров
        s_contours, m_contours, l_contours = self.detect_plants(bitmap, frame_small.shape[:2])

        # 7. Масштабирование контуров обратно к оригинальному размеру
        if self.downscale_factor < 1.0:
            scale = 1.0 / self.downscale_factor
            s_contours = [(cnt * scale).astype(np.int32) for cnt in s_contours]
            m_contours = [(cnt * scale).astype(np.int32) for cnt in m_contours]
            l_contours = [(cnt * scale).astype(np.int32) for cnt in l_contours]

            bitmap_display = cv2.resize(bitmap, (original_rgb.shape[1], original_rgb.shape[0]),
                                        interpolation=cv2.INTER_NEAREST)
        else:
            bitmap_display = bitmap

        # 8. Создание изображения с bounding boxes и количеством растений
        bboxes_img = original_rgb.copy()
        thickness = 2

        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]  # S, M, L
        total_count = 0

        for contours, color in zip([s_contours, m_contours, l_contours], colors):
            for cnt in contours:
                if isinstance(cnt, np.ndarray) and cnt.size > 0 and cnt.shape[0] > 0:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(bboxes_img, (x, y), (x + w, y + h), color, thickness)
                    total_count += 1

        # Добавляем текст с количеством растений
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        font_thickness = 3
        text = f"Total plants: {total_count}"
        text_x = 20
        text_y = 50

        # Тень для текста
        cv2.putText(bboxes_img, text, (text_x + 2, text_y + 2), font, font_scale, (0, 0, 0), font_thickness + 2)
        # Основной текст
        cv2.putText(bboxes_img, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)

        # 9. Конвертируем bitmap в RGB для отображения
        bitmap_rgb = cv2.cvtColor(bitmap_display, cv2.COLOR_GRAY2RGB)

        return original_rgb, index_viz, bitmap_rgb, bboxes_img, total_count

    def visualize_index(self, index):
        """Визуализация цветового индекса в черно-белом формате"""
        index_min, index_max = index.min(), index.max()

        if index_max == index_min:
            index_norm = np.zeros_like(index, dtype=np.uint8)
        else:
            index_norm = ((index - index_min) * 255 / (index_max - index_min)).astype(np.uint8)

        index_gray = cv2.cvtColor(index_norm, cv2.COLOR_GRAY2RGB)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(index_gray, f"Min: {index_min:.3f}", (10, 25), font, 0.6, (0, 0, 0), 3)
        cv2.putText(index_gray, f"Max: {index_max:.3f}", (10, 50), font, 0.6, (0, 0, 0), 3)
        cv2.putText(index_gray, f"Min: {index_min:.3f}", (10, 25), font, 0.6, (255, 255, 255), 2)
        cv2.putText(index_gray, f"Max: {index_max:.3f}", (10, 50), font, 0.6, (255, 255, 255), 2)

        return index_gray