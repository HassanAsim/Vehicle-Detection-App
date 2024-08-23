Introduction:
The Real-Time Vehicle Detection System leverages advancements in artificial intelligence, particularly in computer vision, to identify and classify vehicles in video footage. This project utilizes the YOLO (You Only Look Once) object detection model, known for its high speed and accuracy. The system is designed to detect cars, trucks, motorcycles, buses, and vans in real-time, providing a robust solution for traffic monitoring, autonomous driving, and intelligent transportation systems. The graphical user interface (GUI) allows users to select video files, adjust detection confidence thresholds, and view detection results seamlessly.

Methodology:
1)	Model Initialization:
The YOLOv9 model is loaded from a specified file path.
2)	GUI Development:
A Tkinter-based GUI is developed to facilitate video selection, confidence threshold adjustment, and result display.
3)	Video Processing:
The selected video is processed frame-by-frame. Each frame is passed through the YOLO model to detect vehicles.
4)	Detection and Visualization:
Detected vehicles are filtered based on a confidence threshold. Bounding boxes and labels are drawn around detected vehicles using OpenCV.
5)	Logging: 
Detection results, including the type of vehicle, confidence score, and bounding box coordinates, are logged for further analysis.
6)	Output Generation:
The processed video, with detected vehicles highlighted, is saved to an output file and displayed in real-time.

Project Scope:
The Real-Time Vehicle Detection Application aims to provide a comprehensive solution for detecting and classifying vehicles in various environments. 

The scope includes:

•	Vehicle Detection: Identifying and classifying different types of vehicles such as cars, trucks, motorcycles, buses, and vans.

•	Real-Time Processing: Ensuring that vehicle detection occurs in real-time, suitable for applications in traffic monitoring and autonomous driving.

•	User Interface: Providing a user-friendly interface for video selection, configuration of detection parameters, and visualization of results.

•	Scalability: Designing the system to be scalable, allowing for integration with more advanced models and additional functionalities in the future.

Project Tools:
•	Programming Language: Python
•	IDE: VS Code

Concepts used:
•	Computer Vision: Image and video processing techniques.
•	Deep Learning: Use of convolutional neural networks (CNNs) for object detection.
•	Real-Time Processing: Techniques for handling and processing video frames in real time.
•	OpenCV: Library for computer vision tasks.
•	TensorFlow/PyTorch: Frameworks for implementing and training deep learning models.

Operations Performed:
1)	Video Frame Extraction: Capture and process video frames in real time.
2)	Object Detection: Apply the trained model to detect Vehicles in each frame.
3)	Bounding Box Drawing: Mark detected cars with bounding boxes.
4)	Display: Show the processed video with real-time detections.

Application Snapshots:

GUI:

![image](https://github.com/user-attachments/assets/7553188d-6b40-424d-a953-22e6f5e641a3)

Vehicle Detection:
![image](https://github.com/user-attachments/assets/6d5ca58d-bf01-4a52-9220-96d644871b1c)
![image](https://github.com/user-attachments/assets/3b509ce3-496c-46fa-bfac-3176bd114d69)
![image](https://github.com/user-attachments/assets/714b3b10-18c8-419b-b6d3-097a2eab6c57)
![image](https://github.com/user-attachments/assets/73eeb643-6d2a-47f4-9aed-126a98927f15)
![image](https://github.com/user-attachments/assets/d06e829a-1acb-42ff-9972-ba5c59abb580)
![image](https://github.com/user-attachments/assets/4a741725-62e7-4486-89b9-b588c849e011)






