import sys
import os
import time
import numpy as np
import cv2
import tensorflow as tf
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QFileDialog, QMessageBox, QTreeWidget, QTreeWidgetItem,
    QStyleFactory, QSizePolicy, QFrame, QGridLayout, QHeaderView
)
from PyQt6.QtGui import QImage, QPixmap, QFont, QCloseEvent, QPalette, QColor
from PyQt6.QtCore import Qt, QThread, QObject, pyqtSignal, pyqtSlot, QSize

# --- Constants ---
DEFAULT_MODEL_PATH = "Path to your model.h5"
DEFAULT_LABEL_PATH = "Path to your labels.txt"
CAMERA_INDEX = 0
APP_TITLE = "Qt Real-time Medical Detection"
RESULTS_TITLE = "Detection Results"

# --- Worker Object for Detection Thread ---
class DetectionWorker(QObject):
    frame_ready = pyqtSignal(np.ndarray)
    results_ready = pyqtSignal(list)
    status_update = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    model_load_status = pyqtSignal(bool, str) # Success/fail, message/path
    labels_load_status = pyqtSignal(bool, str) # Success/fail, message/path

    def __init__(self, camera_index=0):
        super().__init__()
        self.camera_index = camera_index
        self._running = False
        self.cap = None
        self.model = None
        self.labels = []
        self.input_shape = None
        self._model_path = ""
        self._label_path = ""

    @pyqtSlot(str)
    def set_model_path(self, path):
        self._model_path = path

    @pyqtSlot(str)
    def set_label_path(self, path):
        self._label_path = path

    @pyqtSlot()
    def load_model(self):
        if not self._model_path or not os.path.exists(self._model_path):
            self.model_load_status.emit(False, "Model path is invalid or not set.")
            return
        try:
            self.status_update.emit("Loading model...")
            self.model = tf.keras.models.load_model(self._model_path)
            try:
                self.input_shape = self.model.input_shape[1:3] # H, W
            except Exception:
                self.input_shape = None
            self.status_update.emit(f"Model loaded. Input shape: {self.input_shape or 'Unknown'}")
            self.model_load_status.emit(True, self._model_path)
        except Exception as e:
            error_msg = f"Error loading model: {e}"
            print(error_msg)
            self.model = None
            self.input_shape = None
            self.status_update.emit("Failed to load model.")
            self.error_occurred.emit(error_msg)
            self.model_load_status.emit(False, error_msg)

    @pyqtSlot()
    def load_labels(self):
        if not self._label_path or not os.path.exists(self._label_path):
            self.labels_load_status.emit(False, "Label path is invalid or not set.")
            return
        try:
            self.status_update.emit("Loading labels...")
            with open(self._label_path, 'r') as f:
                self.labels = [line.strip() for line in f if line.strip()]
            if not self.labels:
                 self.status_update.emit("Label file is empty.")
                 self.labels_load_status.emit(False, "Label file is empty.")
                 return

            self.status_update.emit(f"Labels loaded ({len(self.labels)} classes).")
            self.labels_load_status.emit(True, self._label_path)
        except Exception as e:
            error_msg = f"Error loading labels: {e}"
            print(error_msg)
            self.labels = []
            self.status_update.emit("Failed to load labels.")
            self.error_occurred.emit(error_msg)
            self.labels_load_status.emit(False, error_msg)

    @pyqtSlot()
    def run_detection(self):
        if self.model is None:
            self.error_occurred.emit("Model not loaded. Cannot start detection.")
            return

        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            self.error_occurred.emit(f"Cannot open camera index {self.camera_index}.")
            self.cap = None
            return

        self._running = True
        self.status_update.emit("Detection started.")
        last_update_time = time.time()
        frame_count = 0
        fps = 0

        while self._running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                self.status_update.emit("Error reading frame.")
                time.sleep(0.1)
                continue

            try:
                processed_frame_data = self._preprocess_frame(frame)
                predictions = self.model.predict(processed_frame_data)
                results = self._postprocess_predictions(predictions, frame.shape)
                display_frame = self._draw_results(frame.copy(), results)

                # --- FPS Calculation (Optional) ---
                frame_count += 1
                current_time = time.time()
                elapsed = current_time - last_update_time
                if elapsed >= 1.0:
                    fps = frame_count / elapsed
                    last_update_time = current_time
                    frame_count = 0
                cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

                self.frame_ready.emit(display_frame)
                self.results_ready.emit(results)

            except Exception as e:
                 error_msg = f"Error during detection loop: {e}"
                 print(error_msg)
                 import traceback
                 traceback.print_exc()
                 self.error_occurred.emit(error_msg)
                 # Continue loop or break? Let's try continuing. Use self.stop_detection() to break.


        if self.cap:
            self.cap.release()
        self.cap = None
        self._running = False
        self.status_update.emit("Detection stopped.")
        print("Detection worker loop finished.")


    def _preprocess_frame(self, frame):
        if self.input_shape:
            target_h, target_w = self.input_shape
        else:
            target_h, target_w = 224, 224

        img_resized = cv2.resize(frame, (target_w, target_h))
        img_normalized = img_resized / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)
        return img_batch.astype(np.float32)

    def _postprocess_predictions(self, predictions, original_shape):
        results = []
        detection_threshold = 0.5
        h, w = original_shape[:2]

        if predictions is not None and len(predictions) > 0:
            preds = predictions[0]
            for detection in preds:
                if len(detection) >= 6: # Example: [y1, x1, y2, x2, class_id, score]
                    score = detection[5]
                    if score >= detection_threshold:
                        rel_y1, rel_x1, rel_y2, rel_x2 = detection[0:4]
                        class_id = int(detection[4])
                        abs_x1 = int(rel_x1 * w)
                        abs_y1 = int(rel_y1 * h)
                        abs_x2 = int(rel_x2 * w)
                        abs_y2 = int(rel_y2 * h)
                        results.append({
                            "box": [abs_x1, abs_y1, abs_x2, abs_y2],
                            "class_id": class_id,
                            "score": score
                        })
        return results

    def _draw_results(self, frame, results):
        for result in results:
            abs_x1, abs_y1, abs_x2, abs_y2 = result["box"]
            class_id = result["class_id"]
            score = result["score"]

            if self.labels and 0 <= class_id < len(self.labels):
                label = self.labels[class_id]
            else:
                label = f"Class {class_id}"
            display_text = f"{label}: {score:.2f}"

            cv2.rectangle(frame, (abs_x1, abs_y1), (abs_x2, abs_y2), (0, 255, 0), 2)
            (tw, th), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (abs_x1, abs_y1 - th - 4), (abs_x1 + tw, abs_y1), (0, 255, 0), -1)
            cv2.putText(frame, display_text, (abs_x1, abs_y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        return frame

    @pyqtSlot()
    def stop_detection(self):
        self._running = False


# --- Results Window ---
class ResultsWindow(QWidget):
    window_closed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(RESULTS_TITLE)
        self.setGeometry(150, 150, 450, 350) # x, y, w, h

        self.layout = QVBoxLayout(self)
        self.tree = QTreeWidget(self)
        self.tree.setColumnCount(3)
        self.tree.setHeaderLabels(["Detected Class", "Confidence", "Bounding Box"])
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.header().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)

        self.layout.addWidget(self.tree)
        self.setLayout(self.layout)
        self._is_detection_running = True # Assume starts running

    def set_detection_status(self, running):
        self._is_detection_running = running

    @pyqtSlot(list)
    def update_results(self, results_data):
        self.tree.clear()
        items = []
        for result in results_data:
            class_id = result["class_id"]
            score = result["score"]
            box = result["box"]

            try:
                # Assume self.parent() exists and has 'labels' (fragile design)
                # A better way is to pass labels in or use signals/slots
                main_app = self.parent()
                if main_app and hasattr(main_app, 'get_labels') and main_app.get_labels():
                    labels = main_app.get_labels()
                    if 0 <= class_id < len(labels):
                        class_name = labels[class_id]
                    else:
                        class_name = f"Class {class_id}"
                else:
                     class_name = f"Class {class_id}"
            except Exception:
                 class_name = f"Class {class_id}"


            box_str = f"[{box[0]}, {box[1]}, {box[2]}, {box[3]}]"
            item = QTreeWidgetItem([class_name, f"{score:.3f}", box_str])
            items.append(item)
        self.tree.addTopLevelItems(items)

    def closeEvent(self, event: QCloseEvent):
        if self._is_detection_running:
            event.ignore() # Don't close if detection is running
            self.hide() # Hide instead
            QMessageBox.information(self, "Info", "Results window hidden. It will reappear/close when detection stops.")
        else:
            self.window_closed.emit()
            event.accept()


# --- Main Application Window ---
class MedicalDetectionApp(QMainWindow):
    request_model_load = pyqtSignal()
    request_labels_load = pyqtSignal()
    request_detection_start = pyqtSignal()
    request_detection_stop = pyqtSignal()
    request_set_model_path = pyqtSignal(str)
    request_set_label_path = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.setGeometry(100, 100, 800, 650) # x, y, w, h

        self._model_path = ""
        self._label_path = ""
        self._model_loaded = False
        self._labels_loaded = False
        self._detection_active = False
        self._current_labels = [] # Store loaded labels

        self.setup_ui()
        self.setup_worker_thread()
        self.apply_stylesheet() # Apply custom styling

        self.results_window = None

        # Try loading defaults
        if os.path.exists(DEFAULT_MODEL_PATH):
            self._update_model_path(DEFAULT_MODEL_PATH)
            self.request_model_load.emit()
        if os.path.exists(DEFAULT_LABEL_PATH):
             self._update_label_path(DEFAULT_LABEL_PATH)
             self.request_labels_load.emit()


    def setup_ui(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # --- Controls ---
        control_layout = QGridLayout()

        control_layout.addWidget(QLabel("Model File:"), 0, 0)
        self.model_path_label = QLabel("No model selected")
        self.model_path_label.setWordWrap(True)
        control_layout.addWidget(self.model_path_label, 0, 1)
        self.browse_model_btn = QPushButton("Browse Model")
        self.browse_model_btn.clicked.connect(self._browse_model)
        control_layout.addWidget(self.browse_model_btn, 0, 2)

        control_layout.addWidget(QLabel("Label File:"), 1, 0)
        self.label_path_label = QLabel("No labels selected")
        self.label_path_label.setWordWrap(True)
        control_layout.addWidget(self.label_path_label, 1, 1)
        self.browse_label_btn = QPushButton("Browse Labels")
        self.browse_label_btn.clicked.connect(self._browse_labels)
        control_layout.addWidget(self.browse_label_btn, 1, 2)

        control_layout.setColumnStretch(1, 1) # Path label expands
        main_layout.addLayout(control_layout)

        # --- Video Display ---
        self.video_label = QLabel("Camera Feed - Press Start")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored) # Allow shrinking/expanding
        self.video_label.setFrameShape(QFrame.Shape.Box)
        self.video_label.setFrameShadow(QFrame.Shadow.Sunken)
        self.video_label.setStyleSheet("background-color: #151515; color: #808080; border: 1px solid #404040;")
        self.video_label.setMinimumSize(QSize(320, 240)) # Minimum size
        main_layout.addWidget(self.video_label, 1) # Make video area expand

        # --- Action Buttons ---
        action_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Detection")
        self.start_btn.clicked.connect(self._start_detection)
        self.stop_btn = QPushButton("Stop Detection")
        self.stop_btn.clicked.connect(self._stop_detection)
        self.stop_btn.setEnabled(False)
        action_layout.addWidget(self.start_btn)
        action_layout.addWidget(self.stop_btn)
        main_layout.addLayout(action_layout)

        # --- Status Bar ---
        self.status_bar_label = QLabel("Ready")
        self.status_bar_label.setStyleSheet("padding: 2px; color: #A0A0A0;")
        # Add to QMainWindow's status bar area
        self.statusBar().addWidget(self.status_bar_label, 1)


    def setup_worker_thread(self):
        self.worker_thread = QThread()
        self.detection_worker = DetectionWorker(CAMERA_INDEX)
        self.detection_worker.moveToThread(self.worker_thread)

        # --- Connect Worker Signals to Main Thread Slots ---
        self.detection_worker.frame_ready.connect(self._update_video_display)
        self.detection_worker.results_ready.connect(self._update_results_table)
        self.detection_worker.status_update.connect(self._update_status_bar)
        self.detection_worker.error_occurred.connect(self._handle_error)
        self.detection_worker.model_load_status.connect(self._handle_model_loaded)
        self.detection_worker.labels_load_status.connect(self._handle_labels_loaded)

        # --- Connect Main Thread Request Signals to Worker Slots ---
        self.request_model_load.connect(self.detection_worker.load_model)
        self.request_labels_load.connect(self.detection_worker.load_labels)
        self.request_detection_start.connect(self.detection_worker.run_detection)
        self.request_detection_stop.connect(self.detection_worker.stop_detection)
        self.request_set_model_path.connect(self.detection_worker.set_model_path)
        self.request_set_label_path.connect(self.detection_worker.set_label_path)

        # --- Thread Management ---
        self.worker_thread.started.connect(lambda: print("Worker thread started"))
        self.worker_thread.finished.connect(lambda: print("Worker thread finished"))
        self.worker_thread.start()

    def apply_stylesheet(self):
        # Basic Dark Theme QSS
        qss = """
            QMainWindow {
                background-color: #2E2E2E;
            }
            QWidget { /* Apply to central widget and potentially others */
                background-color: #2E2E2E;
                color: #E0E0E0;
                font-family: Arial;
                font-size: 10pt;
            }
            QLabel {
                background-color: transparent; /* Avoid overriding specific labels */
                color: #E0E0E0;
            }
            QPushButton {
                background-color: #4A4A4A;
                color: #FFFFFF;
                border: 1px solid #5A5A5A;
                padding: 6px 12px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #606060;
            }
            QPushButton:pressed {
                background-color: #3A3A3A;
            }
            QPushButton:disabled {
                background-color: #404040;
                color: #808080;
                border-color: #454545;
            }
            QTreeWidget {
                background-color: #3C3C3C;
                color: #E0E0E0;
                border: 1px solid #505050;
                alternate-background-color: #424242;
            }
            QTreeWidget::item:selected {
                background-color: #5A5A5A;
            }
            QHeaderView::section {
                background-color: #4A4A4A;
                color: #FFFFFF;
                padding: 4px;
                border: 1px solid #606060;
                font-weight: bold;
            }
            QStatusBar {
                background-color: #252525;
            }
            QMessageBox {
                 background-color: #383838; /* Doesn't always work perfectly */
                 color: #E0E0E0;
            }
            QMessageBox QLabel { /* Style label inside message box */
                 background-color: transparent;
                 color: #E0E0E0;
            }
            QMessageBox QPushButton { /* Style buttons inside message box */
                min-width: 60px;
            }
        """
        self.setStyleSheet(qss)
        # Force style update for potentially complex widgets like QMessageBox
        QApplication.setStyle(QStyleFactory.create('Fusion'))

        # Style the results window separately if needed, or ensure it inherits
        # (Inheritance usually works for basic QWidgets)

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Keras Model File", "", "HDF5 files (*.h5);;All files (*.*)")
        if path:
            self._update_model_path(path)
            self.request_model_load.emit() # Request worker to load

    def _browse_labels(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Label File", "", "Text files (*.txt);;All files (*.*)")
        if path:
            self._update_label_path(path)
            self.request_labels_load.emit() # Request worker to load

    def _update_model_path(self, path):
        self._model_path = path
        self.model_path_label.setText(os.path.basename(path))
        self.model_path_label.setToolTip(path)
        self.request_set_model_path.emit(path) # Inform worker

    def _update_label_path(self, path):
        self._label_path = path
        self.label_path_label.setText(os.path.basename(path))
        self.label_path_label.setToolTip(path)
        self.request_set_label_path.emit(path) # Inform worker

    @pyqtSlot(str)
    def _update_status_bar(self, message):
        self.status_bar_label.setText(message)

    @pyqtSlot(bool, str)
    def _handle_model_loaded(self, success, msg_or_path):
        self._model_loaded = success
        if not success:
             self.model_path_label.setText("Load Failed!")
             self.model_path_label.setToolTip(msg_or_path)

    @pyqtSlot(bool, str)
    def _handle_labels_loaded(self, success, msg_or_path):
        self._labels_loaded = success
        if success:
             # Store labels locally if needed by ResultsWindow or drawing logic in main thread
             # This assumes worker keeps its own label list.
             try:
                 with open(msg_or_path, 'r') as f:
                    self._current_labels = [line.strip() for line in f if line.strip()]
             except:
                 self._current_labels = []
        else:
            self.label_path_label.setText("Load Failed!")
            self.label_path_label.setToolTip(msg_or_path)
            self._current_labels = []

    @pyqtSlot(str)
    def _handle_error(self, message):
        QMessageBox.critical(self, "Error", message)
        # Potentially stop detection if error is critical
        if self._detection_active:
             self._stop_detection()


    @pyqtSlot(np.ndarray)
    def _update_video_display(self, cv_img):
        if not self._detection_active: return
        try:
            rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            # Scale pixmap while preserving aspect ratio to fit the label
            pixmap = QPixmap.fromImage(qt_image).scaled(self.video_label.size(),
                                                        Qt.AspectRatioMode.KeepAspectRatio,
                                                        Qt.TransformationMode.SmoothTransformation)
            self.video_label.setPixmap(pixmap)
        except Exception as e:
             print(f"Error updating video display: {e}")
             # Fallback or clear
             self.video_label.setText("Error displaying frame")


    @pyqtSlot(list)
    def _update_results_table(self, results_data):
        if self.results_window and self.results_window.isVisible():
             self.results_window.update_results(results_data)


    def _start_detection(self):
        if self._detection_active:
            return
        if not self._model_loaded:
            QMessageBox.warning(self, "Warning", "Model not loaded successfully. Cannot start.")
            return
        if not self._labels_loaded:
            # Decide if labels are mandatory or optional
             QMessageBox.warning(self, "Warning", "Labels not loaded. Detections will show class indices.")
             # return # Uncomment if labels are mandatory

        self._detection_active = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.browse_model_btn.setEnabled(False)
        self.browse_label_btn.setEnabled(False)
        self.video_label.setText("Starting Camera...") # Clear placeholder

        if self.results_window is None:
            self.results_window = ResultsWindow(self) # Pass self as parent
            self.results_window.window_closed.connect(self._on_results_window_closed_signal)

        self.results_window.set_detection_status(True)
        self.results_window.show()
        self.results_window.raise_() # Bring to front

        self.request_detection_start.emit()


    def _stop_detection(self):
        if not self._detection_active:
            return

        self._detection_active = False
        self.request_detection_stop.emit() # Ask worker to stop gracefully

        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.browse_model_btn.setEnabled(True)
        self.browse_label_btn.setEnabled(True)
        self.video_label.setText("Detection Stopped") # Clear placeholder
        self.video_label.setPixmap(QPixmap()) # Clear image

        if self.results_window:
            self.results_window.set_detection_status(False)
            self.results_window.close() # Allow actual closing now
            # self.results_window = None # Handled by _on_results_window_closed_signal

    def _on_results_window_closed_signal(self):
        print("Results window confirmed closed.")
        self.results_window = None


    def get_labels(self):
         # Method for ResultsWindow to access labels (better alternatives exist)
         return self._current_labels

    def closeEvent(self, event: QCloseEvent):
        print("Main window closing...")
        if self._detection_active:
            self._stop_detection() # Stop worker first

        if self.results_window:
             self.results_window.set_detection_status(False)
             self.results_window.close() # Ensure results window closes

        if self.worker_thread.isRunning():
            print("Requesting worker thread quit...")
            self.worker_thread.quit()
            if not self.worker_thread.wait(3000): # Wait up to 3 seconds
                print("Worker thread did not quit gracefully, terminating.")
                self.worker_thread.terminate()
                self.worker_thread.wait() # Wait after terminate

        print("Exiting application.")
        event.accept()


# --- Run the Application ---
if __name__ == "__main__":
    # Optional: Try to force Fusion style for consistency across platforms
    try:
        QApplication.setStyle(QStyleFactory.create('Fusion'))
    except Exception as e:
        print(f"Could not set Fusion style: {e}")

    app = QApplication(sys.argv)
    main_win = MedicalDetectionApp()
    main_win.show()
    sys.exit(app.exec())
