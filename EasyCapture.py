import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import cv2
import datetime
import logging
from tkinter import messagebox, Tk

class EasyCapture:
    def __init__(self, source, width=1920, height=1080, interval=0.2, save_directory=None):
        self.source = source
        self.width = width
        self.height = height
        self.interval = interval
        self.cap = None
        self.isInit = False
        self.save_all = False
        self.base_dir = os.path.dirname(os.path.abspath(__file__))

        # save_directoryの処理
        if save_directory is None or not os.path.isdir(save_directory):
            self.save_directory = self.base_dir
        else:
            self.save_directory = save_directory

        self.source_type = "unknown"
        self._setup_logging()
        self._open_source()

    def _setup_logging(self):
        log_path = os.path.join(self.base_dir, "EasyCapture.log")
        logging.basicConfig(
            filename=log_path,
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s"
        )
        logging.info("EasyCapture initialized")

    def _show_error(self, message):
        logging.error(message)
        root = Tk()
        root.withdraw()
        messagebox.showerror("Error", message)
        root.destroy()

    def _open_source(self):
        try:
            print("Open source...")
            if isinstance(self.source, int):
                self.cap = cv2.VideoCapture(self.source)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.source_type = "webcam"

            elif isinstance(self.source, str):
                lower_src = self.source.lower()
                video_exts = [".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm"]
                if lower_src.startswith("http") or lower_src.startswith("rtsp"):
                    self.cap = cv2.VideoCapture(self.source)
                    self.source_type = "network"
                elif os.path.splitext(lower_src)[1] in video_exts:
                    self.cap = cv2.VideoCapture(self.source)
                    self.source_type = "video_file"
                else:
                    raise ValueError(f"Unsupported file type: {self.source}")
            else:
                raise ValueError("Invalid source type.")

            if not self.cap or not self.cap.isOpened():
                self._show_error(f"Failed to open video source: {self.source}")
                return

            logging.info(f"Successfully opened {self.source_type}: {self.source}")
            self.isInit = True

        except Exception as e:
            self._show_error(str(e))

    def grab(self):
        if not self.cap or not self.cap.isOpened():
            logging.warning("grab() called but capture not initialized.")
            return
        ret, frame = self.cap.read()
        if not ret:
            logging.warning("Failed to grab frame.")
            return

        now = datetime.datetime.now()
        date_str = now.strftime("%Y%m%d")
        time_str = now.strftime("%Y%m%d_%H%M%S_%f")[:-3]

        save_dir = os.path.join(self.save_directory, date_str)
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, f"{time_str}.jpg")
        cv2.imwrite(filename, frame)
        logging.info(f"Image saved to: {filename}")

    def open_display(self):
        if not self.isInit:
            logging.warning("open_display() called but capture not initialized.")
            return

        logging.info("Display started.")
        if self.source_type == "video_file":
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        save_all_mode = False

        while True:
            ret, frame = self.cap.read()
            if not ret:
                logging.info("No frame to read (end of video or error).")
                break

            # 2キーで連続保存モードONなら“REC”を左上に描画
            if save_all_mode:
                # 表示位置(左上)、フォント、フォントサイズ(0.7)、赤色(BGR: 0,0,255)、太さ(2)
                cv2.putText(frame, "REC", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            cv2.imshow("EasyCapture", frame)

            key = cv2.waitKey(int(self.interval * 1000)) & 0xFF
            if key == ord("q"):
                logging.info("Quit signal received.")
                break
            elif key == ord("1"):
                self.grab()
            elif key == ord("2"):
                save_all_mode = not save_all_mode
                logging.info(f"Save-all mode {'enabled' if save_all_mode else 'disabled'}.")

            if save_all_mode:
                self.grab()

        self.finish()

    def setInterval(self, interval=0.2):
        self.interval = interval
        logging.info(f"Set interval={interval}.")

    def setWidthHeight(self, width=1920, height=1080):
        self.width = width
        self.height = height

        if not self.isInit:
            logging.warning("setWidthHeight() called but capture not initialized.")
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        logging.info(f"Set width={width} and height={height}.")

    def finish(self):
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        logging.info("Capture and display finished.")