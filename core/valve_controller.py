import time
import threading


class ValveController:
    def __init__(self, gpio_pin=7, valve_open_time=0.5, debug=False):
        self.gpio_pin = gpio_pin
        self.valve_open_time = valve_open_time
        self.debug = debug
        self.valve_open = False
        self.valve_lock = threading.Lock()

        if not debug:
            try:
                import OPi.GPIO as GPIO
                GPIO.setboard(GPIO.PCPC2)
                GPIO.setmode(GPIO.BOARD)
                GPIO.setup(self.gpio_pin, GPIO.OUT)
                GPIO.output(self.gpio_pin, GPIO.HIGH)
                self.gpio = GPIO
                print(f"GPIO {gpio_pin} инициализирован")
            except Exception as e:
                print(f"Ошибка GPIO: {e}")
                self.gpio = None
        else:
            print("Режим отладки")
            self.gpio = None

    def open_valve(self):
        with self.valve_lock:
            if self.valve_open:
                return

            self.valve_open = True

            if self.debug:
                print(f"КЛАПАН ОТКРЫТ на {self.valve_open_time} сек")
            else:
                if self.gpio:
                    self.gpio.output(self.gpio_pin, self.gpio.LOW)
                    print(f"КЛАПАН ОТКРЫТ на {self.valve_open_time} сек")

            timer_thread = threading.Thread(target=self._close_after_delay)
            timer_thread.daemon = True
            timer_thread.start()

    def _close_after_delay(self):
        time.sleep(self.valve_open_time)

        with self.valve_lock:
            self.valve_open = False

            if self.debug:
                print("КЛАПАН ЗАКРЫТ")
            else:
                if self.gpio:
                    self.gpio.output(self.gpio_pin, self.gpio.HIGH)
                    print("КЛАПАН ЗАКРЫТ")

    def is_valve_open(self):
        return self.valve_open

    def cleanup(self):
        if self.gpio:
            self.gpio.output(self.gpio_pin, self.gpio.HIGH)
            self.gpio.cleanup()


class CentralStripDetector:
    def __init__(self, strip_height_percent=0.3, min_plant_area=100):
        self.strip_height_percent = strip_height_percent
        self.min_plant_area = min_plant_area

    def check_plants_in_strip(self, plants_contours, frame_shape):
        if not plants_contours:
            return False, None, None, []

        import cv2
        import numpy as np

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
        import cv2
        result = frame.copy()
        color = (0, 0, 255) if has_plants else (0, 255, 0)
        cv2.rectangle(result, (0, strip_top), (result.shape[1], strip_bottom), color, 2)
        text = "pant detect!" if has_plants else "wait plant"
        cv2.putText(result, text, (10, strip_top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return result