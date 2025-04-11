Qt Real-time Medical Detection App 🩺
=====================================

Your Quick Guide

[*(Developed by: Dotmini Software - Tirawat Nantamas dev@dotmini.in.th)*](mailto:dev@dotmini.in.th "null")

✨ What's it Do? (Features)
--------------------------

-   Live Video: Shows you what your computer's camera sees. 📹

-   Pick Your Brain: Lets you browse and load your own Keras/TensorFlow model (`.h5` file). 🧠

-   Know Your Labels: You can also load a text file (`.txt`) that tells the app what the different things your model detects are called. 🏷️

-   Real-time Detection: Once you hit start, it runs your model on the video feed *live*. ⚡

-   See the Results:

    -   Draws boxes around detected objects directly on the video feed. 🖼️

    -   Pops up a separate window listing all the detections with names, confidence scores, and box coordinates. 📊

-   Looks Kinda Sleek: Uses Qt6 with a dark theme. 😎

-   Handles the Heavy Lifting: Runs detection in a separate thread so the app doesn't freeze. 💪

🚀 Getting Started (How to Use)
-------------------------------

1.  📦 Get the Goods: Make sure you have the necessary Python libraries. Open your terminal or command prompt and run:

    ```
    pip install PyQt6 opencv-python tensorflow numpy

    ```

    *`(You might need tensorflow-cpu instead of tensorflow if you don't have a compatible GPU).`*

2.  💾 Save the Code: Grab the Python script (the one ending in `.py`) and save it somewhere (e.g., `qt_medical_detector.py`).

3.  ▶️ Run It! Open your terminal/command prompt again, navigate to where you saved the file, and run:

    ```
    python qt_medical_detector.py

    ```

4.  📂 Load 'Er Up:

    -   Click "Browse Model" and find your `.h5` model file.

    -   Click "Browse Labels" and find your `.txt` label file (make sure the order of names matches what your model outputs!).

    -   Check the status bar at the bottom for loading messages. 💬

5.  ✅ Hit Start: Once loaded, click "Start Detection".

6.  👀 Watch the Magic:

    -   The main window shows the camera feed with detection boxes.

    -   A second "Detection Results" window appears listing finds.

7.  ⏹️ Stop When Done: Click "Stop Detection" to end the process.

🛠️ Super Important: Making it Work with *Your* Model ⚠️
--------------------------------------------------------

Okay, this is key. The code has placeholder bits for handling *your specific model*. You need to edit the Python script (`qt_medical_detector.py`) for it to work correctly. Find the `DetectionWorker` class and modify these methods:

### 1\. ⚙️ `_preprocess_frame(self, frame)`:

-   What it does: Gets the raw camera image ready for your model.

-   Your Job: Change resizing dimensions and normalization methods to match your model's exact input requirements.

### 2\. 🔍 `_postprocess_predictions(self, predictions, original_shape)`:

-   What it does: Takes the raw numbers your model outputs and turns them into useful info (boxes, classes, scores).

-   Your Job (Most Critical!):

    -   Figure out how your model outputs bounding boxes (format, coordinate system).

    -   Extract class IDs and confidence scores correctly.

    -   Filter weak detections.

    -   Crucially: Convert relative coordinates (0-1) to absolute pixel coordinates.

    -   Add Non-Max Suppression (NMS) if needed.

    -   If this isn't right, you'll get no detections, wrong boxes, or maybe even errors! 💥

### 3\. 🖌️ `_draw_results(self, frame, results)`:

-   What it does: Draws the boxes and labels onto the video frame.

-   Your Job: Usually minor changes needed, just ensure the box format used for drawing matches what `_postprocess_predictions` produces.

Seriously, if you don't customize those functions, the app probably won't detect anything correctly!

❓ Quick Troubleshooting Tips
----------------------------

-   No Camera Feed? 📷❌ Check the `CAMERA_INDEX` constant in the script (usually 0, but might be 1 or higher). Make sure your camera is connected and not used by another app.

-   Model/Label Load Failed? 💾❌ Double-check the file paths are correct and the files aren't corrupted. Look for specific error messages in the status bar or console.

-   No Detections Appearing? 🚫 This is almost always an issue in the `_postprocess_predictions` function. Review your logic there carefully! Also check the `detection_threshold`.

-   Wrong Boxes/Labels? ↔️ Again, check `_postprocess_predictions` for coordinate conversion errors. Ensure your label file order matches the model's output.

-   App Freezes or Crashes? 🥶💥 Look for error messages printed in the terminal where you ran the script. This often points to problems in the detection functions or library issues.

🧑‍💻 Developer Info
--------------------

App developed by:

Dotmini Software - Tirawat Nantamas

dev@dotmini.in.th

© 2025 - Have fun detecting! 🎉
