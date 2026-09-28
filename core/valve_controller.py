import ctypes
import ctypes.util
import time
import threading
import cv2
import numpy as np

# ---------- Загрузка libwiringPi.so ----------
def _load_wiringpi():
    candidates = [
        ctypes.util.find_library("wiringPi"),
        "/usr/local/lib/libwiringPi.so",
        "/usr/lib/libwiringPi.so",
    ]
    for path in candidates:
        if path:
            try:
                return ctypes.CDLL(path)
            except OSError:
                continue
    raise RuntimeError(
        "Не найдена libwiringPi.so. Установите WiringOP: "
        "cd ~/wiringOP && sudo ./build && sudo ldconfig"
    )


lib = _load_wiringpi()

# ---------- Прототипы функций ----------
lib.wiringPiSetup.restype     = ctypes.c_int
lib.wiringPiSetupGpio.restype = ctypes.c_int
lib.wiringPiSetupPhys.restype = ctypes.c_int

lib.pinMode.argtypes     = [ctypes.c_int, ctypes.c_int]
lib.pinMode.restype      = None

lib.digitalWrite.argtypes = [ctypes.c_int, ctypes.c_int]
lib.digitalWrite.restype  = None

lib.digitalRead.argtypes = [ctypes.c_int]
lib.digitalRead.restype  = ctypes.c_int


# ---------- Константы ----------
INPUT  = 0
OUTPUT = 1
LOW    = 0
HIGH   = 1


class ValveController:
    def __init__(self, gpio_pin, valve_open_time=0.5, debug=False, pin_mode="wpi"):
        """
        :param gpio_pin:        номер пина
        :param valve_open_time: время открытия клапана, сек
        :param debug:           если True — не трогаем GPIO, только печатаем
        :param pin_mode:        'wpi' | 'gpio' | 'phys'
        """
        self.gpio_pin = gpio_pin
        self.valve_open_time = valve_open_time
        self.debug = debug
        self.valve_open = False
        self.valve_lock = threading.Lock()
        self.gpio_ready = False

        if not debug:
            try:
                if pin_mode == "wpi":
                    rc = lib.wiringPiSetup()
                elif pin_mode == "gpio":
                    rc = lib.wiringPiSetupGpio()
                elif pin_mode == "phys":
                    rc = lib.wiringPiSetupPhys()
                else:
                    raise ValueError(f"Неизвестный pin_mode: {pin_mode}")

                if rc == -1:
                    raise RuntimeError(f"wiringPiSetup({pin_mode}) вернул -1")

                lib.pinMode(self.gpio_pin, OUTPUT)
                lib.digitalWrite(self.gpio_pin, HIGH)   # закрыто
                self.gpio_ready = True
                print(f"GPIO {gpio_pin} ({pin_mode}) инициализирован")
            except Exception as e:
                print(f"Ошибка GPIO: {e}")
        else:
            print("Режим отладки")

    def open_valve(self):
        with self.valve_lock:
            if self.valve_open:
                return
            self.valve_open = True

            if self.gpio_ready:
                lib.digitalWrite(self.gpio_pin, LOW)
            print(f"КЛАПАН ОТКРЫТ на {self.valve_open_time} сек")

            t = threading.Thread(target=self._close_after_delay)
            t.daemon = True
            t.start()

    def _close_after_delay(self):
        time.sleep(self.valve_open_time)

        with self.valve_lock:
            self.valve_open = False
            if self.gpio_ready:
                lib.digitalWrite(self.gpio_pin, HIGH)
            print("КЛАПАН ЗАКРЫТ")

    def is_valve_open(self):
        return self.valve_open

    def cleanup(self):
        if self.gpio_ready:
            lib.digitalWrite(self.gpio_pin, HIGH)
            lib.pinMode(self.gpio_pin, INPUT)


class CentralStripDetector:
    def __init__(self, strip_height_percent=0.3, min_plant_area=100):
        self.strip_height_percent = strip_height_percent
        self.min_plant_area = min_plant_area

    def check_plants_in_strip(self, plants_contours, frame_shape):
        if not plants_contours:
            return False, None, None, []

        h, w = frame_shape[:2]

        strip_height = int(h * self.strip_height_percent)
        strip_top = (h - strip_height) // 2
        strip_bottom = strip_top + strip_height
        strip_center = (strip_top + strip_bottom) // 2

        plants_in_strip = []

        for contour in plants_contours:
            if not isinstance(contour, np.ndarray) or contour.size == 0:
                continue

            area = cv2.contourArea(contour)

            if area < self.min_plant_area:
                continue

            x, y, w_plant, h_plant = cv2.boundingRect(contour)
            plant_center_y = y + h_plant // 2

            if strip_top <= plant_center_y <= strip_bottom:
                plants_in_strip.append({
                    'contour': contour,
                    'center_y': plant_center_y,
                    'area': area,
                    'bbox': (x, y, w_plant, h_plant)
                })

        return len(plants_in_strip) > 0, strip_center, (strip_top, strip_bottom), plants_in_strip

    def draw_central_strip(self, frame, strip_top, strip_bottom, has_plants=False):
        result = frame.copy()
        color = (0, 0, 255) if has_plants else (0, 255, 0)
        cv2.rectangle(result, (0, strip_top), (result.shape[1], strip_bottom), color, 2)
        text = "pant detect!" if has_plants else "wait plant"
        cv2.putText(result, text, (10, strip_top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return result