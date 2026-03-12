# CloudAI - Tkinter Dark Mode Version
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import inference
import ctypes
import time
MAX_VALUE = 100
model = inference.LoadModel()
myappid = "cloudai.classifier.v1"
ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
# bar animation function
def animate_bar(value):
    progress["value"] = 0
    for i in range(value):
        progress["value"] = i
        time.sleep(0.01)
        root.update_idletasks()

# window
root = tk.Tk()
root.title("CloudAI - Cloud Image Classifier")
window_width = 400
window_height = 700

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

x = (screen_width // 2) - (window_width // 2)
y = (screen_height // 2) - (window_height // 2)

root.geometry(f"{window_width}x{window_height}+{x}+{y}")
root.iconbitmap("GUIassets/logo.ico")
root.configure(bg="black")

# ttk style
style = ttk.Style()
style.theme_use("default")

style.configure("white.Horizontal.TProgressbar",
                background="white",
                troughcolor="black",
                bordercolor="black",
                lightcolor="white",
                darkcolor="white")

# logo
logo_img = Image.open("GUIassets/logo_2.jpeg").resize((100,100))
logo_photo = ImageTk.PhotoImage(logo_img)

logo_label = tk.Label(root, image=logo_photo, bg="black")
logo_label.pack()

title_label = tk.Label(root, text="CloudAI", font=("Helvetica",24),
                       fg="white", bg="black")
title_label.pack()

subtitle = tk.Label(root, text="AI Cloud Image Classifier",
                    font=("Helvetica",14),
                    fg="white", bg="black")
subtitle.pack()

# select button
def select_image():

    file_path = filedialog.askopenfilename(
        title="Select Cloud Image",
        filetypes=[("Image Files","*.png *.jpg *.jpeg")]
    )

    if not file_path:
        return

    if not inference.check_image(file_path):
        messagebox.showerror("Error","Please select a valid image file")
        return

    # preview
    img = Image.open(file_path).resize((256,256))
    photo = ImageTk.PhotoImage(img)

    image_label.config(image=photo)
    image_label.image = photo

    # inference
    cloud_type, confidence = model.inference(file_path)
    confidence_percent = int(confidence*100)

    confidence_label.config(text=f"Confidence: {confidence_percent}%")
    animate_bar(confidence_percent)
    cloud_label.config(text=cloud_type)

# button
select_button = tk.Button(root,
                          text="Select Cloud Image",
                          command=select_image,
                          bg="#222222",
                          fg="white",
                          activebackground="#333333",
                          activeforeground="white")
select_button.pack(pady=10)

# preview image
default_img = Image.open("GUIassets/no_cloud_available.jpeg").resize((256,256))
default_photo = ImageTk.PhotoImage(default_img)

image_label = tk.Label(root, image=default_photo, bg="black")
image_label.pack()

# confidence
confidence_label = tk.Label(root,
                            text="Confidence: 0%",
                            fg="white",
                            bg="black")
confidence_label.pack()

# progressbar
progress = ttk.Progressbar(root,
                           style="white.Horizontal.TProgressbar",
                           length=250,
                           maximum=MAX_VALUE)
progress.pack(pady=5)

# cloud type
cloud_title = tk.Label(root,
                       text="Cloud Type:",
                       fg="white",
                       bg="black")
cloud_title.pack()

cloud_label = tk.Label(root,
                       text="N/A",
                       font=("Arial",20),
                       fg="white",
                       bg="black")
cloud_label.pack()

root.mainloop()