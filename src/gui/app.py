import os
import threading
import tkinter as tk
from PIL import Image
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox


class ImageApp:
    def __init__(self, root, model, label_map):
        self.root = root
        self.model = model
        self.label_map = label_map
        
        self.root.title("Main Window")
        self.root.geometry("1000x600")

        self.uploaded_image_path = None
        self.second_window = None
        self.spinner_label = None
        self.spinner_active = False
        self.spinner_dots = 0
        
        self.upload_button = tk.Button(root, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(pady=20)

        self.submit_button = tk.Button(root, text="Submit", command=self.start_background_task)
        self.submit_button.pack(pady=10)

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif"), ("All files", "*.*")]
        )
        if file_path:
            self.uploaded_image_path = file_path
            print(f"Uploaded: {file_path}")

    def start_background_task(self):
        if not self.uploaded_image_path:
            messagebox.showerror("No Image", "Please upload an image first.")
            return

        self.root.withdraw()  

        self.second_window = tk.Toplevel(self.root)
        self.second_window.title("Processing...")
        self.second_window.geometry("1000x600")

        self.spinner_label = tk.Label(self.second_window, text="Processing.", font=("Arial", 24))
        self.spinner_label.pack(pady=100)

        self.spinner_active = True
        self.spinner_dots = 0
        self.rotate_spinner()

        
        threading.Thread(target=self.background_process, daemon=True).start()


    def rotate_spinner(self):
        if not self.spinner_active:
            return

        dots = '.' * (self.spinner_dots + 1)
        self.spinner_label.config(text=f"Processing{dots}", font=("Arial", 24))
        self.spinner_dots = (self.spinner_dots + 1) % 3

        self.root.after(500, self.rotate_spinner)


    def background_process(self):
        img = Image.open(self.uploaded_image_path).convert("RGB")
        scores, predictions = self.model(img)
        
        all_same = all(pred == predictions[0] for pred in predictions)
        if not all_same:
            max_idx = scores.index(max(scores))
            pred = predictions[max_idx]
        else:
            pred = predictions[0]
        pred_label = self.label_map.get(pred, "Unknown")
        original_label = os.path.basename(self.uploaded_image_path).split('.')[0][2:5]
        
        result_text = f"GT: {original_label}    PRED: {pred_label}"
        img = Image.open(self.uploaded_image_path).resize((200, 200))
        img1 = img.copy().resize((100, 100))
        img2 = img.copy().resize((100, 100))

        self.root.after(0, lambda: self.open_second_window(img, img1, img2, result_text))


    def open_second_window(self, main_img, small_img1, small_img2, result_text):
        self.spinner_active = False  

        for widget in self.second_window.winfo_children():
            widget.destroy()

        main_img_tk = ImageTk.PhotoImage(main_img)
        label_main = tk.Label(self.second_window, image=main_img_tk)
        label_main.image = main_img_tk
        label_main.pack(pady=10)

        frame = tk.Frame(self.second_window)
        frame.pack()

        img1_tk = ImageTk.PhotoImage(small_img1)
        label1 = tk.Label(frame, image=img1_tk)
        label1.image = img1_tk
        label1.pack(side=tk.LEFT, padx=5)

        img2_tk = ImageTk.PhotoImage(small_img2)
        label2 = tk.Label(frame, image=img2_tk)
        label2.image = img2_tk
        label2.pack(side=tk.LEFT, padx=5)

        textbox = tk.Entry(self.second_window, justify='center')
        textbox.insert(0, result_text)
        textbox.pack(pady=10)

        back_button = tk.Button(self.second_window, text="Back", command=self.go_back)
        back_button.pack(pady=10)


    def go_back(self):
        if self.second_window:
            self.second_window.destroy()
            self.second_window = None
        self.root.deiconify()
