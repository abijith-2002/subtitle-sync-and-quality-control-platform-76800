"""
Detect text bounding boxes and recognized text in a given video file within a specified time frame using EasyOCR.

USAGE (from command line):
    python detect_text_boxes.py --video path/to/video.mp4 --start 10.0 --end 12.0

- --video: Path to the video file to analyze.
- --start: Start time in seconds.
- --end:   End time in seconds.

OUTPUT:
    Prints and saves (in text_boxes.txt) a list of detected bounding boxes (4 points) and recognized text per box.
Requirements:
    - opencv-python
    - numpy
    - easyocr
"""

import cv2
import numpy as np
import argparse
import json
import os
from typing import List
import easyocr

# PUBLIC_INTERFACE
def extract_frames(video_path: str, start_sec: float, end_sec: float, fps: int = 1) -> List[np.ndarray]:
    """
    Extract frames from video between start_sec and end_sec at the given fps.

    Args:
        video_path: Path to the input video file.
        start_sec:  Start time in seconds.
        end_sec:    End time in seconds.
        fps:        Frames per second to extract.

    Returns:
        List of extracted frames as np.ndarray.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}")
        return []
    duration_sec = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    if end_sec > duration_sec:
        end_sec = duration_sec
    frames = []
    curr_sec = start_sec
    while curr_sec <= end_sec:
        frame_id = int(curr_sec * input_fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        success, frame = cap.read()
        if not success:
            break
        frames.append(frame)
        curr_sec += 1.0 / fps
    cap.release()
    return frames

# PUBLIC_INTERFACE
def detect_text_boxes_with_easyocr(frames: List[np.ndarray], languages=['en'], gpu=False) -> List[dict]:
    """
    For each frame, use EasyOCR to detect all text boxes and recognized texts.

    Args:
        frames: List of video frames (np.ndarray, BGR format as from cv2).
        languages: List of languages for OCR.
        gpu: Whether to use GPU for OCR.

    Returns:
        List of dicts for all detections:
        {
            "frame_idx": int,
            "box": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
            "text": str,
            "conf": float
        }
    """
    reader = easyocr.Reader(languages, gpu=gpu)
    all_results = []
    for idx, frame in enumerate(frames):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            detections = reader.readtext(frame_rgb, detail=1, paragraph=False)
            for det in detections:
                bbox, text, conf = det
                # bbox: list of 4 points (x, y) (polygon)
                all_results.append({
                    "frame_idx": idx,
                    "box": [[int(x), int(y)] for (x, y) in bbox],
                    "text": text,
                    "conf": float(conf)
                })
        except Exception as e:
            print(f"[EasyOCR ERROR on frame {idx}]: {e}")
    return all_results

# PUBLIC_INTERFACE
def detect_text_boxes_in_timeframe(video_path: str, start_sec: float, end_sec: float, fps: int = 1, languages=['en'], gpu=False):
    """
    Extract video frames in given timeframe and detect all text boxes/recognized text using EasyOCR.

    Args:
        video_path: Path to video.
        start_sec, end_sec: Time range in seconds.
        fps: Frames per second to sample.

    Returns:
        List of dicts: frame_idx, box, text, conf
    """
    frames = extract_frames(video_path, start_sec, end_sec, fps=fps)
    results = detect_text_boxes_with_easyocr(frames, languages, gpu)
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect text bounding boxes in a video in a given time frame using EasyOCR.")
    parser.add_argument("--video", required=True, type=str, help="Path to the video file.")
    parser.add_argument("--start", required=True, type=float, help="Start time in seconds.")
    parser.add_argument("--end", required=True, type=float, help="End time in seconds.")
    parser.add_argument("--fps", default=1, type=int, help="Frames per second to sample (default: 1)")
    parser.add_argument("--lang", type=str, default='en', help="Comma-separated OCR language codes (default: en).")
    parser.add_argument("--gpu", action='store_true', help="Use GPU for OCR")
    args = parser.parse_args()

    if not os.path.isfile(args.video):
        print("ERROR: Provided video file does not exist.")
        exit(1)
    if args.end < args.start:
        print("ERROR: End time must be after start time.")
        exit(1)
    language_list = [code.strip() for code in args.lang.split(',') if code.strip()]
    positions = detect_text_boxes_in_timeframe(
        args.video, args.start, args.end, fps=args.fps, languages=language_list, gpu=args.gpu
    )

    # Print all boxes and text as JSON
    print(json.dumps(positions, indent=2, ensure_ascii=False))

    # Print per line, and save to text_boxes.txt
    txtbox_file = os.path.join(os.path.dirname(__file__), "text_boxes.txt")
    try:
        with open(txtbox_file, "w", encoding="utf-8") as outf:
            for entry in positions:
                box = entry["box"]
                text = entry["text"].replace("\n", " ").replace("\r", " ")
                conf = entry["conf"]
                line = f'frame_idx: {entry["frame_idx"]}, box: {box}, conf: {conf:.3f}, text: {text}'
                print(line)
                outf.write(line + "\n")
    except Exception as e:
        print(f"ERROR writing text_boxes.txt: {e}")
