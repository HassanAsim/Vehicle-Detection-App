import cv2
import logging
from ultralytics import YOLO
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from threading import Thread
from PIL import Image, ImageTk
import torch
import gc

# Initialize logging
logging.basicConfig(filename='vehicle_detection.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Check for CUDA availability and set device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEVICE == 'cuda':
    torch.backends.cudnn.benchmark = True
    logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    logging.info("Using CPU")
    print("Using CPU")

# Load YOLOv8 model
yolo_model_path = './models/yolov9m.pt'
model = YOLO(yolo_model_path)
model.to(DEVICE)

# Define colors for different labels
label_colors = {
    'car': (0, 255, 0),        # Green
    'truck': (255, 165, 0),      # Orange
    'motorcycle': (0, 0, 255), # Red
    'bus': (255, 255, 0),      # Cyan
    'van': (255, 0, 0)       # Blue
}

# Function to detect vehicles
def detect_vehicles(frame, model, conf_threshold=0.5):
    # Convert frame to CUDA tensor if available
    if DEVICE == 'cuda':
        frame = torch.from_numpy(frame).to(DEVICE)
    
    results = model(frame, stream=True)  # Enable streaming mode for better GPU memory management
    detections = []
    
    for result in results:
        filtered_boxes = [box for box in result.boxes if box.conf[0] >= conf_threshold]
        result.boxes = filtered_boxes
        detections.append(result)
    
    # Clear CUDA cache periodically
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
        gc.collect()
    
    return detections

# Video processing thread
def process_video(video_path, output_path, conf_threshold):
    cap = cv2.VideoCapture(video_path)
    
    # Enable CUDA backend if available
    if DEVICE == 'cuda':
        cap.set(cv2.CAP_PROP_CUDA_DEVICE, 0)
    
    # Set buffer size and frame dropping
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Create named window and set it to normal (allows resizing)
    cv2.namedWindow('Vehicle Detection', cv2.WINDOW_NORMAL)
    
    # Get the screen resolution
    screen_width = cv2.getWindowImageRect('Vehicle Detection')[2]
    screen_height = cv2.getWindowImageRect('Vehicle Detection')[3]
    
    # Move window to top-left corner and make it fullscreen
    cv2.moveWindow('Vehicle Detection', 0, 0)
    cv2.setWindowProperty('Vehicle Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    frame_count = 0
    frame_skip = 1
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        detections = detect_vehicles(frame, model, conf_threshold)
        
        for detection in detections:
            boxes = detection.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = model.names[cls]
                if label in label_colors:
                    color = label_colors[label]
                    label_text = f'{label.capitalize()} {conf:.2f}'
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(frame, (x1, y1 - text_height - baseline - 1), (x1 + text_width, y1), color, thickness=cv2.FILLED)
                    cv2.putText(frame, label_text, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    logging.info(f'Detected {label} with confidence {conf:.2f} at ({x1}, {y1}), ({x2}, {y2})')

        out.write(frame)
        
        # Display frame in fullscreen
        cv2.imshow('Vehicle Detection', frame)
        
        # Press 'q' to quit or 'f' to toggle fullscreen
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('f'):  # Toggle fullscreen with 'f' key
            is_fullscreen = cv2.getWindowProperty('Vehicle Detection', cv2.WND_PROP_FULLSCREEN)
            if is_fullscreen == cv2.WINDOW_FULLSCREEN:
                cv2.setWindowProperty('Vehicle Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            else:
                cv2.setWindowProperty('Vehicle Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # Final cleanup
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
        gc.collect()

# GUI for video selection and configuration
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Vehicle Detection Application")
        self.root.geometry("600x450")

        self.video_path = ""
        self.output_path = "./Outputs/vehicles_detected_output.mp4"
        self.conf_threshold = tk.DoubleVar(value=0.5)

        self.create_widgets()

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text="Confidence Threshold:").pack(pady=5)
        
        self.threshold_label = ttk.Label(main_frame, text=f"{self.conf_threshold.get():.2f}")
        self.threshold_label.pack(pady=5)

        self.conf_slider = ttk.Scale(main_frame, from_=0, to_=1, orient=tk.HORIZONTAL, variable=self.conf_threshold, command=self.update_threshold_label)
        self.conf_slider.pack(fill=tk.X, padx=20)

        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=10)

        ttk.Button(button_frame, text="Select Video", command=self.select_video).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Start Detection", command=self.start_detection).pack(side=tk.LEFT, padx=5)

        self.video_name_label = ttk.Label(main_frame, text="No video selected")
        self.video_name_label.pack(pady=5)

        self.video_label = ttk.Label(main_frame, text="")
        self.video_label.pack(pady=10)

    def update_threshold_label(self, event):
        self.threshold_label.config(text=f"{self.conf_threshold.get():.2f}")

    def select_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi")])
        if self.video_path:
            self.video_name_label.config(text=f"Selected Video: {self.video_path.split('/')[-1]}")
            self.show_video_preview()
        else:
            self.video_name_label.config(text="No video selected")

    def show_video_preview(self):
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        cap.release()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            img.thumbnail((300, 300))
            img_tk = ImageTk.PhotoImage(image=img)
            self.video_label.config(image=img_tk, text="")
            self.video_label.image = img_tk
        else:
            self.video_label.config(text="Error loading video")

    def start_detection(self):
        if not self.video_path:
            messagebox.showerror("Error", "No video selected!")
        else:
            Thread(target=process_video, args=(self.video_path, self.output_path, self.conf_threshold.get())).start()

root = tk.Tk()
app = App(root)
root.mainloop()
