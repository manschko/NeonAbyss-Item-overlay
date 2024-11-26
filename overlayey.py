import tkinter as tk

def create_overlay(text):
    root = tk.Tk()
    root.overrideredirect(True)  # Removes window decorations
    root.attributes("-topmost", True)  # Keeps the window on top of all others
    root.attributes("-transparentcolor", "white")  # Makes the background color transparent

    # Create a label to display text
    label = tk.Label(root, text=text, font=('Arial', 20), fg='black', bg='white')
    # label.pack(expand=True)

    # Make the window full screen and transparent to clicks
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.geometry(f"{screen_width}x{screen_height}+0+0")
    root.config(bg='white')

    root.mainloop()

# Text to display on overlay
overlay_text = "This is an overlay!"
create_overlay(overlay_text)
