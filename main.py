import tkinter as tk
from src.gui.app import ImageApp



def main():
    root = tk.Tk()
    app = ImageApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
