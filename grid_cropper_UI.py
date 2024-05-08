import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os

class ImageCroppingAppEnhanced:
    def __init__(self, root):
        self.root = root
        self.root.title("Enhanced Image Cropping Tool")
        self.root.state('zoomed')

        # Button frame
        button_frame = tk.Frame(root, padx=10, pady=10)
        button_frame.pack(side=tk.LEFT, fill=tk.Y)

        self.load_button = tk.Button(button_frame, text="Load Image", command=self.load_image)
        self.load_button.pack(fill=tk.X)

        self.delete_all_button = tk.Button(button_frame, text="Delete All", command=lambda: self.confirm_action(self.delete_all))
        self.delete_all_button.pack(fill=tk.X)

        self.delete_last_button = tk.Button(button_frame, text="Delete Last", command=lambda: self.confirm_action(self.delete_last))
        self.delete_last_button.pack(fill=tk.X)

        self.save_all_button = tk.Button(button_frame, text="Save All", command=self.save_all)
        self.save_all_button.pack(fill=tk.X)

        # Canvas for displaying images
        self.canvas = tk.Canvas(root, bg='white')
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.bind('<MouseWheel>', self.on_mouse_wheel)
        self.canvas.bind('<ButtonPress-2>', self.start_pan)
        self.canvas.bind('<ButtonRelease-2>', self.stop_pan)
        self.canvas.bind('<B2-Motion>', self.pan_image)

        # Scrollbars for the canvas
        self.h_scroll = tk.Scrollbar(root, orient='horizontal', command=self.canvas.xview)
        self.h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.v_scroll = tk.Scrollbar(root, orient='vertical', command=self.canvas.yview)
        self.v_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.canvas.configure(xscrollcommand=self.h_scroll.set, yscrollcommand=self.v_scroll.set)

        self.rect_id = None
        self.rect_size = (128, 128)
        self.image = None
        self.image_id = None
        self.image_path = None
        self.thumbnails = []
        self.labels = []

        # For panning
        self._drag_data = {"x": 0, "y": 0}

        # Bind mouse events
        self.canvas.bind('<Motion>', self.move_rect)
        self.canvas.bind('<Button-1>', self.crop_image)

    def on_mouse_wheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def start_pan(self, event):
        self.canvas.scan_mark(event.x, event.y)

    def stop_pan(self, event):
        self._drag_data["x"] = 0
        self._drag_data["y"] = 0

    def pan_image(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)
        self._drag_data["x"] = event.x
        self._drag_data["y"] = event.y

    def confirm_action(self, action):
        if messagebox.askyesno("Confirmation", "Are you sure?"):
            action()
            
    def load_image(self):
        filepath = filedialog.askopenfilename()
        if not filepath:
            return

        try:
            self.image = Image.open(filepath)
            self.tk_image = ImageTk.PhotoImage(self.image)

            if self.image_id:
                self.canvas.delete(self.image_id)

            self.image_id = self.canvas.create_image(0, 0, image=self.tk_image, anchor='nw')
            self.canvas.config(scrollregion=(0, 0, self.image.width, self.image.height))

            self.thumbnails.clear()
            self.labels.clear()
            self.delete_all_bounding_boxes()

            # Only update self.image_path if a new image is successfully loaded
            self.image_path = filepath
            self.rect_id = self.canvas.create_rectangle(0, 0, self.rect_size[0], self.rect_size[1], outline='red')
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {e}")

    def move_rect(self, event):
        if not self.image:
            return

        canvas_coords = (self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))
        x0 = max(0, min(canvas_coords[0], self.image.width - self.rect_size[0]))
        y0 = max(0, min(canvas_coords[1], self.image.height - self.rect_size[1]))
        self.canvas.coords(self.rect_id, x0, y0, x0 + self.rect_size[0], y0 + self.rect_size[1])

    def crop_image(self, event):
        if not self.image or len(self.thumbnails) >= 30:
            return

        canvas_coords = (self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))
        x1, y1, x2, y2 = self.canvas.coords(self.rect_id)
        cropped_img = self.image.crop((x1, y1, x2, y2)).convert('L')
        tk_thumbnail = ImageTk.PhotoImage(cropped_img)
        self.thumbnails.append((tk_thumbnail, (x1, y1, x2, y2)))  # Ensure this is a tuple
        tag = f"rect{len(self.thumbnails)}"
        self.canvas.create_rectangle(x1, y1, x2, y2, outline='red', tags=tag)
        label = self.canvas.create_text(x1 + 64, y1 + 64, text=str(len(self.thumbnails)), fill='red', tags=tag)
        self.labels.append((tag, label))
        self.refresh_thumbnails()

    def delete_all_bounding_boxes(self):
        for tag, label in self.labels:
            self.canvas.delete(tag)
            self.canvas.delete(label)
        self.labels.clear()

    def refresh_thumbnails(self):
        for widget in self.canvas.find_withtag('thumbnail'):
            self.canvas.delete(widget)

        x_offset = 20
        y_offset = 100
        column = 0
        for idx, (thumb, coords) in enumerate(self.thumbnails):  # Corrected access
            if idx % 6 == 0 and idx > 0:
                column += 1
                y_offset = 100
            self.canvas.create_image(x_offset + column * (self.rect_size[0] + 10), y_offset, image=thumb, anchor='nw', tags='thumbnail')
            self.canvas.create_text(x_offset + column * (self.rect_size[0] + 10) + 64, y_offset + 64, text=str(idx+1), fill='red', tags='thumbnail')
            y_offset += self.rect_size[1] + 10

    def delete_all(self):
        self.thumbnails.clear()
        self.refresh_thumbnails()
        self.delete_all_bounding_boxes()

    def delete_last(self):
        if self.thumbnails:
            self.thumbnails.pop()
            last_tag, last_label = self.labels.pop()
            self.canvas.delete(last_tag)
            self.canvas.delete(last_label)
            self.refresh_thumbnails()

    def save_all(self):
        save_directory = filedialog.askdirectory()
        if not save_directory:
            messagebox.showinfo("Save Canceled", "Save operation canceled.")
            return

        if not self.image_path:
            messagebox.showerror("Error", "No image loaded. Please load an image first.")
            return

        try:
            for idx, (_, coords) in enumerate(self.thumbnails):
                x1, y1, x2, y2 = coords
                cropped_image = self.image.crop((x1, y1, x2, y2)).convert('L')
                filename = f"{os.path.basename(self.image_path)}_crop_{idx+1}_{x1}_{y1}.jpg"
                cropped_img_path = os.path.join(save_directory, filename)
                cropped_image.save(cropped_img_path, 'JPEG')
            messagebox.showinfo("Success", "All images have been saved successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save image: {e}")
            

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageCroppingAppEnhanced(root)
    root.mainloop()
