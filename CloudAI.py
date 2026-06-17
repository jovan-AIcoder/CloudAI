import PySimpleGUI as sg
from PIL import Image
import inference
import io
import numpy as np
import time

MAX_VALUE = 100
model = inference.LoadModel()

current_image = None


# =====================================
# Convert image file to bytes
# =====================================
def image_to_bytes(path, size=None):
    img = Image.open(path)

    if size:
        img = img.resize(size)

    bio = io.BytesIO()
    img.save(bio, format="PNG")

    return bio.getvalue()


# =====================================
# Convert numpy array to bytes
# =====================================
def array_to_bytes(array):

    img = Image.fromarray(
        np.uint8(array)
    )

    bio = io.BytesIO()
    img.save(bio, format="PNG")

    return bio.getvalue()


# =====================================
# Layout
# =====================================
sg.theme("DarkBlack")

layout = [

    [
        sg.Image(
            data=image_to_bytes(
                "GUIassets/logo_2.jpeg",
                (100, 100)
            )
        )
    ],

    [
        sg.Text(
            "CloudAI",
            font=("Arial", 24),
            text_color="white"
        )
    ],

    [
        sg.Text(
            "AI Cloud Image Classifier",
            font=("Arial", 12),
            text_color="white"
        )
    ],

    [
        sg.Button(
            "Select Cloud Image",
            key="-SELECT-"
        ),

        sg.Button(
            "Grad-CAM Analysis",
            key="-GRADCAM-",
            disabled=True
        )
    ],

    [
        sg.Image(
            data=image_to_bytes(
                "GUIassets/no_cloud_available.jpeg",
                (256, 256)
            ),
            key="-IMAGE-"
        )
    ],

    [
        sg.Text(
            "Confidence: 0%",
            key="-CONFIDENCE-",
            text_color="white"
        )
    ],

    [
        sg.ProgressBar(
            MAX_VALUE,
            orientation="h",
            size=(30, 20),
            key="-PROGRESS-",
            bar_color=("white", "black")
        )
    ],

    [
        sg.Text(
            "Cloud Type:",
            text_color="white"
        )
    ],

    [
        sg.Text(
            "N/A",
            key="-CLOUD-TYPE-",
            font=("Arial", 20),
            text_color="white"
        )
    ]
]


# =====================================
# Main Window
# =====================================
window = sg.Window(
    "CloudAI - Cloud Image Classifier",
    layout,
    size=(400, 650),
    element_justification="center",
    icon="GUIassets/logo.ico",
    finalize=True
)


# =====================================
# Event Loop
# =====================================
while True:

    event, values = window.read()

    if event == sg.WINDOW_CLOSED:
        break

    # =================================
    # Select image
    # =================================
    elif event == "-SELECT-":

        file_path = sg.popup_get_file(
            "Select Cloud Image",
            file_types=(
                ("Image Files",
                 "*.png;*.jpg;*.jpeg"),
            ),
            icon='GUIassets/logo.ico'
        )

        if not file_path:
            continue

        if not inference.check_image(file_path):

            sg.popup_error(
                "Please select a valid image file."
            )

            continue

        current_image = file_path

        # Enable Grad-CAM button
        window["-GRADCAM-"].update(
            disabled=False
        )

        # Update preview
        window["-IMAGE-"].update(
            data=image_to_bytes(
                file_path,
                (256, 256)
            )
        )

        # Inference
        cloud_type, confidence = (
            model.inference(
                file_path
            )
        )

        confidence_percent = int(
            confidence * 100
        )

        # Animate progress bar
        window["-PROGRESS-"].update(0)

        for i in range(
                confidence_percent + 1):

            window["-PROGRESS-"].update(i)
            window.refresh()
            time.sleep(0.01)

        window["-CONFIDENCE-"].update(
            f"Confidence: {confidence_percent}%"
        )

        window["-CLOUD-TYPE-"].update(
            cloud_type
        )

    # =================================
    # Grad-CAM Analysis
    # =================================
    elif event == "-GRADCAM-":

        if current_image is None:
            continue

        cam = model.GradCAM(
            current_image
        )

        cam_layout = [

            [
                sg.Text(
                    "Grad-CAM Analysis",
                    font=("Helvetica", 20),
                    text_color="white"
                )
            ],

            [
                sg.Image(
                    data=array_to_bytes(
                        cam
                    )
                )
            ],

            [
                sg.Button(
                    "Close"
                )
            ]
        ]

        cam_window = sg.Window(
            "CloudAI - Grad-CAM Analysis",
            cam_layout,
            modal=True,
            element_justification="center",
            icon="GUIassets/logo.ico"
        )

        while True:

            e, v = cam_window.read()

            if e in (
                    sg.WINDOW_CLOSED,
                    "Close"
            ):
                break

        cam_window.close()


window.close()