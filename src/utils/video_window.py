import threading
from queue import Queue

import numpy as np
from PyQt5.QtCore import Qt, QTime, pyqtSignal, QObject
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsTextItem, \
    QGridLayout
from numpy._typing import NDArray


class VideoWindow(QWidget):
    closed = pyqtSignal()

    def __init__(self, num_cameras: int):
        super().__init__()
        self.setWindowTitle('Live Tracking')

        self.grid = QGridLayout()
        self.setLayout(self.grid)

        self.camera_widgets: list[CameraWidget] = []
        self.workers: list[CameraWorker] = []
        self.threads: list[threading.Thread] = []

        for i in range(num_cameras):
            widget = CameraWidget(i + 1, parent=self)
            self.camera_widgets.append(widget)

        self.update_grid_layout(1, 2)

    def update_grid_layout(self, rows, cols):
        for i in reversed(range(self.grid.count())):
            self.grid.itemAt(i).widget().setParent(None)

        index = 0
        for row in range(rows):
            for col in range(cols):
                if index < len(self.camera_widgets):
                    self.grid.addWidget(self.camera_widgets[index], row, col)
                    index += 1

    def update_camera_frame(self, camera_id: int, frame_id: int, frame: NDArray[np.uint8]):
        self.camera_widgets[camera_id].update_frame(frame_id, frame)

    def closeEvent(self, event):
        for worker in self.workers:
            worker.running = False
        for thread in self.threads:
            if thread.is_alive():
                thread.join()
        self.closed.emit()
        super().closeEvent(event)


class CameraWidget(QWidget):
    def __init__(self, camera_id: int, parent=None):
        super().__init__(parent)
        self.camera_id = camera_id

        self.view = QGraphicsView()
        self.scene = QGraphicsScene()
        self.view.setScene(self.scene)

        self.pixmap = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap)

        self.title_text = QGraphicsTextItem(f"Camera #{camera_id}")
        self.title_text.setDefaultTextColor(Qt.white)
        self.title_text.setPos(10, 10)
        self.scene.addItem(self.title_text)

        self.fps_label = QGraphicsTextItem("FPS: 0")
        self.fps_label.setDefaultTextColor(Qt.green)
        self.fps_label.setPos(10, 40)
        self.scene.addItem(self.fps_label)

        layout = QVBoxLayout()
        layout.addWidget(self.view)
        self.setLayout(layout)

        self.prev_frame_id = 0
        self.prev_time = QTime.currentTime()
        self.fps = 0

    def update_fps(self, frame_id: int) -> None:
        frame_count = frame_id - self.prev_frame_id
        current_time = QTime.currentTime()
        elapsed = self.prev_time.msecsTo(current_time)

        self.prev_frame_id = frame_id
        self.prev_time = current_time
        self.fps = frame_count / (elapsed / 1000.0)
        self.fps_label.setPlainText(f"FPS: {self.fps:.1f}")

    def update_frame(self, frame_id: int, frame: NDArray[np.uint8]) -> None:
        h, w, c = frame.shape
        bytes_per_line = c * w
        q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.pixmap.setPixmap(QPixmap.fromImage(q_img))

        self.update_fps(frame_id)


class CameraWorker(QObject):
    def __init__(self, camera_id: int, parent=None):
        super().__init__(parent)
        self.camera_id = camera_id
        self.running = True
        self.queue = Queue(maxsize=2)
        self.queue_lock = threading.Lock()
