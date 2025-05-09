import numpy as np
from PyQt5.QtCore import Qt, QTime, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsTextItem
from numpy._typing import NDArray


class VideoWindow(QWidget):
    closed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Live Tracking')

        self.view = QGraphicsView()
        self.scene = QGraphicsScene()
        self.view.setScene(self.scene)

        self.pixmap = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap)

        self.fps_label = QGraphicsTextItem("FPS: 0")
        self.fps_label.setDefaultTextColor(Qt.green)
        self.fps_label.setPos(10, 10)
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

    def update_frame(self, frame: NDArray[np.uint8]) -> None:
        h, w, c = frame.shape
        bytes_per_line = c * w
        q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.pixmap.setPixmap(QPixmap.fromImage(q_img))

    def closeEvent(self, event):
        self.closed.emit()
        super().closeEvent(event)
