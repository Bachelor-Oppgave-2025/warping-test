# Ball Tracking System

A computer vision system for tracking balls on a pool/billiards table from video.

## Features

- Automatic table corner detection (works with any camera angle)
- Automatic orientation detection (portrait/landscape)
- Perspective transformation to top-down view
- Ball detection and centroid tracking
- Side-by-side or stacked display of original and warped views

## Requirements

- Python 3.7+
- OpenCV (cv2)
- NumPy

## Installation

Install dependencies:
```bash
pip install opencv-python numpy
```

## Usage
Make a `videos` folder and place your videos in the `videos/` folder, then run:
```bash
python main.py
```

Select a video when prompted (by number or file path).

## Controls

- SPACE: Pause/Resume
- ESC: Quit

## File Structure

- `main.py` - Main entry point
- `detection.py` - Table and ball detection
- `warping.py` - Perspective transformation
- `tracking.py` - Centroid tracking
- `display.py` - Display and visualization
- `videos/` - Video files directory
