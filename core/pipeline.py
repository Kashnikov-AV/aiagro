"""Конвейер обработки видеопотока (pipeline).

Архитектура: кадр проходит через цепочку независимых шагов (stages), каждый шаг
получает и возвращает словарь-контекст `ctx` с именованными артефактами:

    ctx = {"frame": ndarray(BGR), ...}
    шаг 1: DownscaleStage      -> ctx["small"], ctx["scale"]
    шаг 2: VegetationIndex     -> ctx["index"]        (ExG/ExR/ExGR/CIVE)
    шаг 3: VisualizeIndex      -> ctx["index_viz"]    (для виджета №2)
    шаг 4: OtsuThreshold       -> ctx["binary"]
    шаг 5: Morphology          -> ctx["bitmap"]       (для виджета №3)
    шаг 6: DetectContours      -> ctx["contours_s/m/l"]
    шаг 7: Classify            -> ctx["plants"]       (центр, площадь, размер)
    шаг 8: ROI / StopLine      -> ctx["in_roi"]       (GOB: центральная полоса;
                                                       GOG: линия останова между камерами)
    шаг 9: DrawOverlays        -> ctx["annotated"]    (для виджета №4)
    шаг 10: Actuator           -> ctx["actuated"]     (клапан GOB / стоп-сигнал GOG)

Пайплайн конфигурируется списком шагов — режимы GOB и GOG отличаются только
набором и параметрами шагов. Любой шаг можно заменить или вставить новый без
изменения остальных (открыто-закрытый принцип).

Литература по индексам растительности (реализованы формулы):
 - ExG  = 2g - r - b        (Woebbecke et al., 1995), где r,g,b = R,G,B/(R+G+B).
 - ExR  = 1.4r - g            (Woebbecke et al., 1995)
 - ExGR = 3g - 2.4r - b        (Adamsen et al., 2000)
 - CIVE = 0.441R - 0.811G + 0.385B + 18.787 (Kjaering, 1995)
 - ExNVI= 1.3g - nvii (See et al., 2002); NVII = (2NIR-R)/(2NIR+G) — требует NIR.
Для сегментации «сорняк vs культурное растение» в row-crop системах стандартный
приём — геометрическое исключение междурядий (Bird et al., 2007; Garcia y Garcia
et al., 2009), что у нас соответствует центральной полосе (GOB) и линии останова
(GOG). Пороговая бинаризация — Otsu (1979).
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
#  Контекст и базовые классы
# --------------------------------------------------------------------------- #
@dataclass
class PipelineContext:
    """Именованные артефакты, передаваемые между шагами пайплайна."""
    frame: np.ndarray                      # входной кадр BGR
    small: np.ndarray | None = None        # уменьшенный кадр (float32 RGB)
    scale: float = 1.0                     # масштаб small -> frame
    index: np.ndarray | None = None        # карта вегетационного индекса
    binary: np.ndarray | None = None       # бинарная маска (uint8 0/255)
    bitmap: np.ndarray | None = None       # после морфологии
    contours: dict[str, list] = field(default_factory=dict)   # s/m/l
    plants: list[dict] = field(default_factory=list)          # классифицированные
    in_roi: list[dict] = field(default_factory=list)          # растения в зоне
    annotated: np.ndarray | None = None    # кадр с разметкой (RGB для Qt)
    result: dict = field(default_factory=dict)                # сводка/метрики
    stop: bool = False                     # досрочно завершить цепочку


class Stage(ABC):
    """Базовый шаг пайплайна. Наследуйтесь или используйте FunctionStage."""
    name: str = "stage"

    @abstractmethod
    def process(self, ctx: PipelineContext) -> PipelineContext:
        ...


class FunctionStage(Stage):
    """Шаг-обёртка над обычной функцией: FunctionStage("my", fn).

    fn(ctx) -> ctx | None (None означает «ctx не изменён»).
    Удобно добавлять экспериментальные шаги без создания класса.
    """

    def __init__(self, name: str, fn: Callable[[PipelineContext], PipelineContext | None]):
        self.name = name
        self._fn = fn

    def process(self, ctx):
        out = self._fn(ctx)
        return ctx if out is None else out


class VideoPipeline:
    """Упорядоченная цепочка шагов. Потокобезопасен (по одному экземпляру на поток)."""

    def __init__(self, stages: list[Stage] | None = None, name: str = "pipeline"):
        self.name = name
        self._stages: list[Stage] = list(stages or [])
        self._lock = threading.Lock()

    # -- конструирование -----------------------------------------------------
    def add(self, stage: Stage, before: str | None = None) -> "VideoPipeline":
        """Добавить шаг в конец (или перед шагом с именем `before`)."""
        with self._lock:
            if before:
                for i, s in enumerate(self._stages):
                    if s.name == before:
                        self._stages.insert(i, stage)
                        return self
                raise ValueError(f"stage '{before}' not found")
            self._stages.append(stage)
        return self

    def remove(self, name: str) -> "VideoPipeline":
        with self._lock:
            self._stages = [s for s in self._stages if s.name != name]
        return self

    def replace(self, name: str, stage: Stage) -> "VideoPipeline":
        with self._lock:
            self._stages = [stage if s.name == name else s for s in self._stages]
        return self

    def names(self) -> list[str]:
        return [s.name for s in self._stages]

    # -- выполнение ----------------------------------------------------------
    def run(self, frame_bgr: np.ndarray) -> PipelineContext:
        ctx = PipelineContext(frame=frame_bgr)
        for stage in self._stages:
            try:
                result = stage.process(ctx)
                if result is not None:  # FunctionStage может вернуть None
                    ctx = result
            except Exception as e:  # шаг не должен ронять приложение
                print(f"[{self.name}] stage '{stage.name}' error: {e}")
            if ctx.stop:
                break
        return ctx


# --------------------------------------------------------------------------- #
#  Встроенные шаги
# --------------------------------------------------------------------------- #
class ToFloatRGBStage(Stage):
    """BGR uint8 -> RGB float32 (нормированные каналы для индексов)."""
    name = "to_rgb"

    def process(self, ctx):
        rgb = cv2.cvtColor(ctx.frame, cv2.COLOR_BGR2RGB).astype(np.float32)
        ctx.small = rgb
        ctx.scale = 1.0
        return ctx


class DownscaleStage(Stage):
    def __init__(self, factor: float = 0.5):
        self.factor = factor

    name = "downscale"

    def process(self, ctx):
        rgb = cv2.cvtColor(ctx.frame, cv2.COLOR_BGR2RGB).astype(np.float32)
        if self.factor < 1.0:
            h, w = rgb.shape[:2]
            rgb = cv2.resize(rgb, (int(w * self.factor), int(h * self.factor)),
                             interpolation=cv2.INTER_LINEAR)
            ctx.scale = 1.0 / self.factor
        else:
            ctx.scale = 1.0
        ctx.small = rgb
        return ctx


class VegetationIndexStage(Stage):
    """Экстракция вегетационного индекса (см. docstring модуля)."""
    name = "veg_index"

    def __init__(self, kind: str = "exg"):
        self.kind = kind.lower()

    def process(self, ctx):
        img = ctx.small
        R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        s = R + G + B + 1e-7
        if self.kind == "exg":
            ctx.index = 2 * G / s - R / s - B / s
        elif self.kind == "exr":
            ctx.index = 1.4 * R / s - G / s
        elif self.kind == "exgr":
            ctx.index = 3 * G / s - 2.4 * R / s - B / s
        elif self.kind == "cive":
            ctx.index = 0.441 * R - 0.811 * G + 0.385 * B + 18.787
        elif self.kind == "excess_green":  # избыточный зелёный без нормировки
            ctx.index = 2 * G - R - B
        else:
            raise ValueError(f"unknown index '{self.kind}'")
        return ctx


class OtsuThresholdStage(Stage):
    """Нормализация индекса + порог Оцу; для exr/cive инверсия."""
    name = "otsu"

    def __init__(self, invert: bool | None = None):
        self.invert = invert

    def process(self, ctx):
        idx = ctx.index
        lo, hi = idx.min(), idx.max()
        norm = np.zeros_like(idx, dtype=np.uint8) if hi == lo else \
            ((idx - lo) * 255 / (hi - lo)).astype(np.uint8)
        _, binary = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        invert = self.invert
        if invert is None:
            invert = False
        if invert:
            binary = cv2.bitwise_not(binary)
        ctx.binary = binary
        return ctx


class MorphologyStage(Stage):
    name = "morphology"

    def __init__(self, kernel_size: int = 3, iterations: int = 1):
        self.kernel = np.ones((kernel_size, kernel_size), np.uint8)
        self.iterations = iterations

    def process(self, ctx):
        m = cv2.morphologyEx(ctx.binary, cv2.MORPH_CLOSE, self.kernel,
                             iterations=self.iterations)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, self.kernel, iterations=self.iterations)
        ctx.bitmap = cv2.dilate(m, self.kernel, iterations=1)
        return ctx


class ContourStage(Stage):
    """Поиск внешних контуров с фильтрацией по минимальной площади."""
    name = "contours"

    def __init__(self, min_area_factor: float = 0.0001):
        self.min_area_factor = min_area_factor

    def process(self, ctx):
        h, w = ctx.bitmap.shape[:2]
        min_area = h * w * self.min_area_factor
        contours, _ = cv2.findContours(ctx.bitmap, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_TC89_L1)
        kept = [c for c in contours if cv2.contourArea(c) > min_area]
        # приводим к координатам оригинального кадра
        kept = [(c * ctx.scale).astype(np.int32) for c in kept]
        ctx.contours = {"all": kept, "s": [], "m": [], "l": []}
        return ctx


class SizeClassifyStage(Stage):
    """Разбиение растений на S/M/L по площади (аналог исходного PlantDetector)."""
    name = "classify"

    def __init__(self, small_factor: float = 0.0001, medium_scale: float = 9,
                 large_scale: float = 4):
        self.small_factor = small_factor
        self.medium_scale = medium_scale
        self.large_scale = large_scale

    def process(self, ctx):
        h_img, w_img = ctx.frame.shape[:2]
        small_s = h_img * w_img * self.small_factor
        medium_s = small_s * self.medium_scale
        large_s = medium_s * self.large_scale
        plants = []
        buckets = {"s": [], "m": [], "l": []}
        for cnt in ctx.contours.get("all", []):
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            size = "l" if area > large_s else "m" if area > medium_s else "s"
            buckets[size].append(cnt)
            plants.append({"contour": cnt, "area": area, "size": size,
                           "bbox": (x, y, w, h),
                           "center": (x + w // 2, y + h // 2)})
        ctx.contours.update(buckets)
        ctx.plants = plants
        ctx.result["total"] = len(plants)
        ctx.result["counts"] = {k: len(v) for k, v in buckets.items()}
        return ctx


class CentralStripROIStage(Stage):
    """GOB: горизонтальная полоса в центре кадра (культурный ряд)."""
    name = "roi_strip"

    def __init__(self, strip_height_percent: float = 0.3, min_area: float = 0):
        self.percent = strip_height_percent
        self.min_area = min_area

    def process(self, ctx):
        h = ctx.frame.shape[0]
        sh = int(h * self.percent)
        top = (h - sh) // 2
        bottom = top + sh
        ctx.result["strip_bounds"] = (top, bottom)
        ctx.in_roi = [p for p in ctx.plants
                      if top <= p["center"][1] <= bottom and p["area"] >= self.min_area]
        return ctx


class StopLineROIStage(Stage):
    """GOG: вертикальная «линия останова» — момент, когда объект камеры
    (растение/сорняк) пересекает заданную X-позицию (например, точку между
    двумя камерами, где расположен исполнительный орган)."""
    name = "roi_stipline"

    def __init__(self, x_ratio: float = 0.5, hysteresis_px: int = 20):
        self.x_ratio = x_ratio
        self.hyst = hysteresis_px

    def process(self, ctx):
        w = ctx.frame.shape[1]
        line_x = int(w * self.x_ratio)
        ctx.result["line_x"] = line_x
        ctx.in_roi = [p for p in ctx.plants
                      if abs(p["center"][0] - line_x) <= self.hyst]
        return ctx


class DrawStage(Stage):
    """Отрисовка рамок, зоны интереса и счётчика. Результат — RGB (для QImage)."""
    name = "draw"

    COLORS = {"s": (0, 200, 0), "m": (255, 128, 0), "l": (0, 0, 255)}

    def __init__(self, draw_original: bool = True):
        self.draw_original = draw_original

    def process(self, ctx):
        base = cv2.cvtColor(ctx.frame, cv2.COLOR_BGR2RGB) if self.draw_original \
            else np.zeros_like(cv2.cvtColor(ctx.frame, cv2.COLOR_BGR2RGB))
        out = base.copy()
        for p in ctx.plants:
            x, y, w, h = p["bbox"]
            color = self.COLORS[p["size"]]
            cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        bounds = ctx.result.get("strip_bounds")
        if bounds:
            top, bottom = bounds
            hit = len(ctx.in_roi) > 0
            color = (255, 0, 0) if hit else (0, 255, 0)
            cv2.rectangle(out, (0, top), (out.shape[1], bottom), color, 2)
            text = "plant detect!" if hit else "wait plant"
            cv2.putText(out, text, (10, max(top - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        line_x = ctx.result.get("line_x")
        if line_x is not None:
            hit = len(ctx.in_roi) > 0
            color = (255, 0, 0) if hit else (0, 255, 0)
            cv2.line(out, (line_x, 0), (line_x, out.shape[0]), color, 2)
        total = ctx.result.get("total", len(ctx.plants))
        cv2.putText(out, f"plants: {total}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 5)
        cv2.putText(out, f"plants: {total}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        ctx.annotated = out
        return ctx


class ValveActuationStage(Stage):
    """Актуатор: открывает клапан при наличии растений в зоне.

    controller — любой объект с open_valve()/is_valve_open() (ValveController),
    либо колбэк. Шаг не блокирует поток: ValveController сам закрывает клапан
    по таймеру.
    """
    name = "actuator"

    def __init__(self, controller, only_large: bool = True, cooldown_s: float = 0.0):
        self.controller = controller
        self.only_large = only_large
        self.cooldown = cooldown_s
        self._last_fire = 0.0

    def process(self, ctx):
        import time
        targets = [p for p in ctx.in_roi
                   if (p["size"] == "l") or not self.only_large]
        now = time.time()
        if targets and not self.controller.is_valve_open() \
                and now - self._last_fire >= self.cooldown:
            self.controller.open_valve()
            self._last_fire = now
            ctx.result["actuated"] = True
        return ctx


class SignalStage(Stage):
    """Вызывает callback(result_dict) — мост из worker-потока в GUI (Qt signal)."""
    name = "signal"

    def __init__(self, callback: Callable[[dict], None]):
        self.callback = callback

    def process(self, ctx):
        try:
            self.callback(dict(ctx.result))
        except Exception as e:
            print(f"[signal stage] callback error: {e}")
        return ctx


class VisualizeIndexStage(Stage):
    """Черно-белая карта индекса с подписями min/max (виджет №2 в GOB)."""
    name = "viz_index"

    def process(self, ctx):
        idx = ctx.index
        lo, hi = idx.min(), idx.max()
        norm = np.zeros_like(idx, dtype=np.uint8) if hi == lo else \
            ((idx - lo) * 255 / (hi - lo)).astype(np.uint8)
        vis = cv2.cvtColor(norm, cv2.COLOR_GRAY2RGB)
        if ctx.scale != 1.0:
            vis = cv2.resize(vis, (ctx.frame.shape[1], ctx.frame.shape[0]),
                             interpolation=cv2.INTER_NEAREST)
        font = cv2.FONT_HERSHEY_SIMPLEX
        for text, y in ((f"Min: {lo:.3f}", 25), (f"Max: {hi:.3f}", 50)):
            cv2.putText(vis, text, (10, y), font, 0.6, (0, 0, 0), 3)
            cv2.putText(vis, text, (10, y), font, 0.6, (255, 255, 255), 2)
        ctx.result["index_viz"] = vis
        return ctx


class BitmapViewStage(Stage):
    """Бинарная маска в RGB-формате (виджет №3 в GOB)."""
    name = "viz_bitmap"

    def process(self, ctx):
        bmp = ctx.bitmap
        if ctx.scale != 1.0:
            bmp = cv2.resize(bmp, (ctx.frame.shape[1], ctx.frame.shape[0]),
                             interpolation=cv2.INTER_NEAREST)
        ctx.result["bitmap_rgb"] = cv2.cvtColor(bmp, cv2.COLOR_GRAY2RGB)
        return ctx


# --------------------------------------------------------------------------- #
#  Готовые конфигурации режимов
# --------------------------------------------------------------------------- #
def build_gob_pipeline(valve_controller=None, index_type: str = "exg",
                       downscale: float = 0.5) -> VideoPipeline:
    """Green-on-Brown: сегментация субстрата, индексы, центральная полоса, клапан.

    Этапы соответствуют научному подходу: вегетационный индекс -> Otsu ->
    морфология -> контуры -> геометрическая ROI (ряд) -> актуатор.
    """
    stages: list[Stage] = [
        DownscaleStage(downscale),
        VegetationIndexStage(index_type),
        VisualizeIndexStage(),
        OtsuThresholdStage(invert=index_type in ("exr", "cive")),
        MorphologyStage(),
        BitmapViewStage(),
        ContourStage(),
        SizeClassifyStage(),
        CentralStripROIStage(strip_height_percent=0.3, min_area=2000),
    ]
    if valve_controller is not None:
        stages.append(ValveActuationStage(valve_controller, only_large=True))
    stages.append(DrawStage())
    return VideoPipeline(stages, name="gob")


def build_gog_pipeline(index_type: str = "exg", downscale: float = 0.5,
                       x_ratio: float = 0.5) -> VideoPipeline:
    """Green-on-Green: оба фона зелёные — индекс работает хуже, поэтому
    упор на геометрию: контуры -> линия останова между камерами.
    Отличается от GOB отсутствием клапана/визуализаций индекса и своей ROI."""
    stages: list[Stage] = [
        DownscaleStage(downscale),
        VegetationIndexStage(index_type),
        OtsuThresholdStage(invert=index_type in ("exr", "cive")),
        MorphologyStage(kernel_size=5),
        ContourStage(min_area_factor=0.0003),
        SizeClassifyStage(),
        StopLineROIStage(x_ratio=x_ratio),
        DrawStage(),
    ]
    return VideoPipeline(stages, name="gog")
