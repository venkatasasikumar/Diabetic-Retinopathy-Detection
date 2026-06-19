from tkinter import *
from tkinter import ttk, filedialog, messagebox
import numpy as np
import cv2
from numpy.random import randn
from keras.models import model_from_json, load_model
from PIL import Image, ImageTk
from threading import Thread

# ================== MAIN WINDOW ==================
main = Tk()
main.title("Diabetic Retinopathy Detection and Severity Grading from Fundus Images Using Hybrid GAN-CNN Framework")
main.geometry("1350x800")
main.configure(bg="#f0f2f5")

# ================== GLOBALS ==================
gan_model = None
predict_model = None
preview_window = None
main_img_tk = None
dataset_path = None

# ================== UTILS ==================
def update_progress(val, msg=""):
    progress['value'] = val
    status_lbl.config(text=msg)
    main.update()

def generate_latent_points(latent_dim, n_samples):
    return randn(latent_dim * n_samples).reshape(n_samples, latent_dim)

def getPrediction(img):
    img = img.reshape(1, 32, 32, 3)
    preds = predict_model.predict(img, verbose=0)
    labels = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
    return labels[np.argmax(preds)]

# ================== GAN IMAGE POPUP ==================
def displayGanImagesPopup(X):
    """Display 5 random GAN grayscale images with predictions in a popup"""
    global preview_window
    if preview_window and preview_window.winfo_exists():
        preview_window.destroy()

    preview_window = Toplevel(main)
    preview_window.title("GAN Predicted Images")
    preview_window.configure(bg="#f0f2f5")

    preview_window.images = []

    for i in range(5):
        index = np.random.randint(0, len(X))
        img = X[index]
        # Normalize [-1,1] -> 0-255
        img_norm = ((img + 1)/2 * 255).clip(0,255).astype(np.uint8)
        gray_img = np.mean(img_norm, axis=2).astype(np.uint8)
        pil_img = Image.fromarray(gray_img).resize((180,180))
        img_tk = ImageTk.PhotoImage(pil_img)

        result = getPrediction(img)

        lbl_frame = Frame(preview_window, bg="white", bd=2, relief=RIDGE)
        lbl_frame.grid(row=0, column=i, padx=10, pady=10)
        lbl_img = Label(lbl_frame, image=img_tk, bg="white")
        lbl_img.pack()
        Label(lbl_frame, text=result, bg="white", fg="#333",
              font=("Segoe UI",10,"bold")).pack(pady=5)

        preview_window.images.append(img_tk)  # keep reference

    Button(preview_window, text="Close", command=preview_window.destroy).grid(row=1, columnspan=5, pady=10)

# ================== GAN MODEL ==================
def ganModelThread():
    global gan_model
    text.insert(END, "Loading GAN model...\n")
    text.update()
    update_progress(10, "Loading GAN model...")
    try:
        gan_model = load_model('model/generator_model_080.h5')
        text.insert(END, "GAN model loaded successfully\n")
        text.update()
        update_progress(50, "Generating GAN images...")
        latent_points = generate_latent_points(200, 200)
        X = gan_model.predict(latent_points, verbose=0)
        text.insert(END, f"Generated Images Shape: {X.shape}\n")
        text.update()
        update_progress(100, "GAN ready")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load GAN model: {e}")
        text.insert(END, "GAN model loading failed\n")
        text.update()
        update_progress(0, "GAN failed")

def ganModel():
    Thread(target=ganModelThread).start()

# ================== PREDICTION MODEL ==================
def predictModelThread():
    global predict_model
    text.insert(END, "Loading prediction model...\n")
    text.update()
    update_progress(10, "Loading prediction model...")
    try:
        with open('model/train.json', "r") as json_file:
            predict_model = model_from_json(json_file.read())
        predict_model.load_weights("model/train.h5")
        text.insert(END, "Prediction model loaded successfully\n")
        text.update()
        update_progress(100, "Prediction model ready")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load prediction model: {e}")
        text.insert(END, "Prediction model loading failed\n")
        text.update()
        update_progress(0, "Prediction model failed")

def predictModel():
    Thread(target=predictModelThread).start()

# ================== GAN PREDICTION ==================
def predictSeverityThread():
    if gan_model is None or predict_model is None:
        messagebox.showerror("Error", "Load both GAN and Prediction models first")
        return

    text.insert(END, "Generating GAN images for prediction...\n")
    text.update()
    update_progress(20, "Generating latent points...")
    latent_points = generate_latent_points(200, 200)
    update_progress(50, "Generating images...")
    X = gan_model.predict(latent_points, verbose=0)
    update_progress(80, "Predicting severity...")

    # Show 5 images in popup
    main.after(0, lambda: displayGanImagesPopup(X))

    update_progress(100, "GAN prediction completed")
    text.insert(END, "GAN prediction completed\n")
    text.update()

def predictSeverity():
    Thread(target=predictSeverityThread).start()

# ================== SINGLE IMAGE ==================
def uploadAndPredict():
    global preview_window, main_img_tk
    if predict_model is None:
        messagebox.showerror("Error", "Load prediction model first")
        return
    file_path = filedialog.askopenfilename(
        filetypes=[('Image Files', '*.jpg *.png *.jpeg *.bmp')]
    )
    if not file_path:
        return

    pil_img = Image.open(file_path).convert('RGB')
    img = np.array(pil_img)
    img_resized = cv2.resize(img, (32,32))
    result = getPrediction(img_resized)

    text.insert(END, f"{file_path}\nPrediction: {result}\n\n")
    text.update()

    if preview_window and preview_window.winfo_exists():
        preview_window.destroy()

    preview_window = Toplevel(main)
    preview_window.title("Prediction Result")
    display_img = pil_img.resize((400,400))
    img_tk = ImageTk.PhotoImage(display_img)
    Label(preview_window, image=img_tk).pack(padx=10,pady=10)
    preview_window.image = img_tk
    Label(preview_window, text=f"Prediction: {result}", font=("Segoe UI",14,"bold")).pack(pady=5)
    Button(preview_window, text="Close", command=preview_window.destroy).pack(pady=5)

    main_img_tk = ImageTk.PhotoImage(display_img.resize((300,300)))
    img_panel.config(image=main_img_tk)

# ================== DATASET UPLOAD ==================
def uploadDataset():
    global dataset_path
    dataset_path = filedialog.askdirectory(initialdir=".")
    if dataset_path:
        text.insert(END, f"Fundus dataset loaded: {dataset_path}\n")
        text.update()
        update_progress(100, "Dataset loaded successfully")
    else:
        text.insert(END, "No folder selected.\n")
        text.update()
        update_progress(0, "Dataset not loaded")

# ================== EXIT ==================
def closeApp():
    main.destroy()

# ================== UI ==================
sidebar = Frame(main, bg="#1e3a5f", width=280)
sidebar.pack(side=LEFT, fill=Y)
Label(sidebar, text="DR Detection System", bg="#1e3a5f", fg="white",
      font=("Segoe UI", 20, "bold"), pady=20).pack()

def on_enter(btn): btn.config(bg="#3e5c8a")
def on_leave(btn): btn.config(bg="#334f73")
def side_btn(txt, cmd):
    b = Button(sidebar, text=txt, command=cmd, bg="#334f73", fg="white",
               font=("Segoe UI",12,"bold"), relief=FLAT, pady=12)
    b.pack(fill=X, padx=20, pady=10)
    b.bind("<Enter>", lambda e: on_enter(b))
    b.bind("<Leave>", lambda e: on_leave(b))
    return b

side_btn("📂 Upload Fundus Dataset", uploadDataset)
side_btn("🧠 Load GAN Model", ganModel)
side_btn("📊 Load Prediction Model", predictModel)
side_btn("🧬 GAN Generate & Predict", predictSeverity)
side_btn("🖼 Upload Image & Predict", uploadAndPredict)
side_btn("❌ Exit", closeApp)

# Main content frame
content = Frame(main, bg="#f0f2f5")
content.pack(side=RIGHT, fill=BOTH, expand=True, padx=20, pady=20)
Label(content, text="Diabetic Retinopathy Detection", bg="#f0f2f5", fg="#111827",
      font=("Segoe UI",22,"bold")).pack(anchor=W, pady=(0,10))
Label(content, text="and Severity Grading from Fundus Images Using Hybrid GAN-CNN Framework",
      bg="#f0f2f5", fg="#555", font=("Segoe UI",12)).pack(anchor=W, pady=(0,20))

# Progress bar and status
progress = ttk.Progressbar(content, length=450)
progress.pack(anchor=W, pady=5)
status_lbl = Label(content, text="Ready", bg="#f0f2f5", fg="#555", font=("Segoe UI",10))
status_lbl.pack(anchor=W, pady=(0,10))

# Output text
card = Frame(content, bg="white", bd=2, relief=RIDGE)
card.pack(fill=X, pady=10)
text = Text(card, height=10, font=("Consolas",10), bg="white", fg="#111827", relief=FLAT)
text.pack(side=LEFT, fill=BOTH, expand=True, padx=10, pady=10)
Scrollbar(card, command=text.yview).pack(side=RIGHT, fill=Y)

# Image preview
img_card = Frame(content, bg="white", bd=2, relief=RIDGE)
img_card.pack(pady=10)
Label(img_card, text="Image Preview", bg="white", fg="#111827", font=("Segoe UI",12,"bold")).pack(pady=10)
img_panel = Label(img_card, bg="#e5e7eb", width=300, height=300)
img_panel.pack(padx=20, pady=10)

main.mainloop()
