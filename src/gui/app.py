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
        
        button_font = ("Arial", 14)
        button_width = 20  
        button_pady_outer = 10 
        button_ipady_inner = 5 
        
        button_frame = tk.Frame(root)
        button_frame.pack(pady=30, expand=True) 

        self.upload_button = tk.Button(button_frame, text="Upload Image", command=self.upload_image,
                                       font=button_font, width=button_width,
                                       pady=button_ipady_inner)
        self.upload_button.pack(pady=button_pady_outer)

        self.submit_button = tk.Button(button_frame, text="Submit", command=self.start_background_task,
                                       font=button_font, width=button_width,
                                       pady=button_ipady_inner)
        self.submit_button.pack(pady=button_pady_outer)

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

        
        base_font_size = 12  
        title_font = ("Arial", base_font_size + 2, "bold")
        text_font = ("Arial", base_font_size)

        
        top_images_frame = tk.Frame(self.second_window)
        top_images_frame.pack(pady=10, padx=10, fill="x", expand=True)

        
        original_image_container = tk.Frame(top_images_frame)
        original_image_container.pack(side=tk.LEFT, padx=10, expand=True, fill="both")

        label_original_title = tk.Label(original_image_container, text="Original Image", font=title_font)
        label_original_title.pack(pady=(0, 5))

        main_img_tk = ImageTk.PhotoImage(main_img)
        label_main_img = tk.Label(original_image_container, image=main_img_tk)
        label_main_img.image = main_img_tk  
        label_main_img.pack(expand=True, fill="both")

        
        
        eye_detection_container = tk.Frame(top_images_frame)
        eye_detection_container.pack(side=tk.RIGHT, padx=10, expand=True, fill="both")

        label_eye_detection_title = tk.Label(eye_detection_container, text="Eye position detection", font=title_font)
        label_eye_detection_title.pack(pady=(0, 5))
        
        
        eye_detection_img_tk = ImageTk.PhotoImage(main_img) 
        label_eye_detection_img = tk.Label(eye_detection_container, image=eye_detection_img_tk)
        label_eye_detection_img.image = eye_detection_img_tk  
        label_eye_detection_img.pack(expand=True, fill="both")

        
        segmentation_container_frame = tk.Frame(self.second_window)
        segmentation_container_frame.pack(pady=10, padx=10, fill="x", expand=True)

        label_segmentation_title = tk.Label(segmentation_container_frame, text="Iris Segmentation", font=title_font)
        label_segmentation_title.pack(pady=(0, 5)) 

        segmentation_images_frame = tk.Frame(segmentation_container_frame)
        segmentation_images_frame.pack() 

        small_img1_tk = ImageTk.PhotoImage(small_img1)
        label_small1 = tk.Label(segmentation_images_frame, image=small_img1_tk)
        label_small1.image = small_img1_tk  
        label_small1.pack(side=tk.LEFT, padx=5, expand=True, fill="both")

        small_img2_tk = ImageTk.PhotoImage(small_img2)
        label_small2 = tk.Label(segmentation_images_frame, image=small_img2_tk)
        label_small2.image = small_img2_tk  
        label_small2.pack(side=tk.LEFT, padx=5, expand=True, fill="both")

        
        textbox_frame = tk.Frame(self.second_window) 
        textbox_frame.pack(pady=10, padx=10, fill="x")
        
        textbox = tk.Entry(textbox_frame, justify='center', font=text_font)
        textbox.insert(0, result_text)
        
        
        textbox.pack(fill="x", expand=True, ipady=5) 

        
        back_button = tk.Button(self.second_window, text="Back", command=self.go_back, font=text_font)
        back_button.pack(pady=(10, 20)) 


    def go_back(self):
        if self.second_window:
            self.second_window.destroy()
            self.second_window = None
        self.root.deiconify()
