# 🚗 Vehicle Detection & Counting Using YOLOv11 + OpenCV

A real-time vehicle detection and counting system built in my **free time** using **YOLOv11** and **OpenCV**, designed to process **video files** and provide live vehicle count analytics.

## 📽️ Demo
This project supports **video input only** (not images). It detects and counts vehicles crossing user-defined lines in either **UP** or **DOWN** directions.

## 🚀 Features

- 🔍 Detects multiple vehicle types: **Cars, Motorcycles, Buses, Trucks**
- 🧠 Uses **YOLOv11** for accurate object detection
- 🎯 Line-crossing logic to count vehicles moving **up** or **down**
- 🖱️ Mouse-based UI to draw counting zones
- 📊 Annotated output video with real-time stats
- 🧼 Clean and readable Python code

## 🛠️ Tech Stack

- [YOLOv11 by Ultralytics](https://github.com/ultralytics/ultralytics)
- OpenCV
- Python 3

## 📂 How to Use

1. Clone the repo:
   ```bash
   git clone https://github.com/Yasssh2123/Smart_Vehicle_Counting.git
   cd Smart_Vehicle_Counting
2. Install the requirements:
   ```bash
   pip3 install -r requirement.txt
3. Run the Code :
   ```bash
   python3 Main.py 
After you run the code, you will be prompted to select two points for drawing each of the counting lines. First, draw the "UP" line to count vehicles moving up, and then draw the "DOWN" line to count vehicles moving down.

      
