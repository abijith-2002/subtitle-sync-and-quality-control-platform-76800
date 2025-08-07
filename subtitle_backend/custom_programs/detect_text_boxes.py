"""
Detect text box positions in a given video file within a specified time frame using logic inspired by FrameWhisperer.

USAGE (from command line):
    python detect_text_boxes.py --video path/to/video.mp4 --start 10.0 --end 12.0

- --video: Path to the video file to analyze.
- --start: Start time in seconds.
- --end:   End time in seconds.

OUTPUT:
    Prints a JSON array of detected text box bounding boxes with (x, y, w, h).

Requirements:
    - opencv-python
    - numpy
    - (Recommended: Tesseract OCR for real-world text detection; this script uses contour-based detection as a proxy for FrameWhisperer's method.)
"""

import cv2
import numpy as np
import argparse
import json
import os
from typing import List, Tuple

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
def detect_text_regions(frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Detects rectangular text region candidates in an image.

    Args:
        frame: Image as an ndarray (BGR).

    Returns:
        List of (x, y, w, h) bounding boxes for each detected region.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, threshed = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Morphology to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    morph = cv2.dilate(threshed, kernel, iterations=2)
    morph = cv2.erode(morph, np.ones((3, 3), np.uint8), iterations=1)

    # Find contours (external only)
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Heuristic: Ignore very small boxes and very wide/short ones that are unlikely text lines
        if w > 40 and h > 15 and w < frame.shape[1] * 0.95 and h < frame.shape[0] * 0.4:
            bboxes.append((x, y, w, h))
    return bboxes

# PUBLIC_INTERFACE
def detect_text_boxes_in_timeframe(video_path: str, start_sec: float, end_sec: float, fps: int = 1):
    """
    Detect all text region bounding boxes in the frames between start_sec and end_sec and extract the text within each box using EasyOCR.

    Args:
        video_path: Path to video.
        start_sec, end_sec: Time range in seconds.

    Returns:
        List of dicts containing "frame_idx", "box": [x, y, w, h], "text": <detected_text>
    """
    frames = extract_frames(video_path, start_sec, end_sec, fps=fps)
    results = []
    # Instantiate the EasyOCR Reader (English only for speed, modify as needed)
    reader = easyocr.Reader(['en'], gpu=False)
    for idx, frame in enumerate(frames):
        boxes = detect_text_regions(frame)
        for box in boxes:
            x, y, w, h = box
            crop = frame[y:y+h, x:x+w]

            # Check crop size for being too small (EasyOCR fails on tiny images)
            if crop.shape[0] < 15 or crop.shape[1] < 15:
                # Skip too-tiny regions (likely noise, or EasyOCR won't work)
                continue

            # Optionally resize very small crops to a reasonable size for OCR
            target_min_size = 50
            if crop.shape[0] < target_min_size or crop.shape[1] < target_min_size:
                scale_h = max(target_min_size / crop.shape[0], 1)
                scale_w = max(target_min_size / crop.shape[1], 1)
                scale = max(scale_h, scale_w)
                new_w = max(int(crop.shape[1] * scale), target_min_size)
                new_h = max(int(crop.shape[0] * scale), target_min_size)
                crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

            # EasyOCR expects RGB
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

            # Extract text with EasyOCR
            try:
                dets = reader.readtext(crop_rgb, detail=0, paragraph=True)
            except Exception as e:
                # Save error for debugging
                dets = []
                text = f"[EasyOCR Error: {e}]"
                results.append({
                    "frame_idx": idx,
                    "box": list(box),
                    "text": text
                })
                continue

            # Join results if multiple lines/words detected
            text = " ".join(dets).strip()
            results.append({
                "frame_idx": idx,
                "box": list(box),
                "text": text
            })
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect text boxes in a video in a given time frame.")
    parser.add_argument("--video", required=True, type=str, help="Path to the video file.")
    parser.add_argument("--start", required=True, type=float, help="Start time in seconds.")
    parser.add_argument("--end", required=True, type=float, help="End time in seconds.")
    parser.add_argument("--fps", default=1, type=int, help="Frames per second to sample (default: 1)")
    args = parser.parse_args()

    if not os.path.isfile(args.video):
        print("ERROR: Provided video file does not exist.")
        exit(1)
    if args.end < args.start:
        print("ERROR: End time must be after start time.")
        exit(1)
    positions = detect_text_boxes_in_timeframe(args.video, args.start, args.end, fps=args.fps)

    # Print the position and detected text as JSON
    print(json.dumps(positions, indent=2))
    # Also print detected text per box, and log any empty or error results for diagnosis
    for entry in positions:
        print(f'frame_idx: {entry["frame_idx"]}, box: {entry["box"]}, text: "{entry["text"]}"')
        # Diagnostic: print warning for empty or obviously erroneous detection results
        if entry["text"] == "" or entry["text"].startswith("[EasyOCR Error"):
            print(f"[DIAG] Empty/OCR error: box={entry['box']}")

    # Optional: Save diagnostic crops for visual debug (uncomment if needed)
    # for i, entry in enumerate(positions):
    #     if entry['text'] == "" or entry["text"].startswith("[EasyOCR Error"):
    #         x, y, w, h = entry['box']
    #         frame_idx = entry["frame_idx"]
    #         out_dir = os.path.join(os.path.dirname(__file__), "ocr_debug_crops")
    #         os.makedirs(out_dir, exist_ok=True)
    #         frame = extract_frames(args.video, args.start, args.end, fps=args.fps)[frame_idx]
    #         crop = frame[y:y+h, x:x+w]
    #         file_out = os.path.join(out_dir, f"frame{frame_idx}_box_{x}_{y}_{w}_{h}.png")
    #         cv2.imwrite(file_out, crop)

    # Write the detected positions and text to text_boxes.txt as plain text, one line per box
    txtbox_file = os.path.join(os.path.dirname(__file__), "text_boxes.txt")
    try:
        with open(txtbox_file, "w") as outf:
            for entry in positions:
                # New format: frame_idx: 3, box: [x, y, w, h], text: Detected Text
                line = f'frame_idx: {entry["frame_idx"]}, box: {entry["box"]}, text: {entry["text"].replace(chr(10), " ").replace(chr(13), " ")}'
                outf.write(line + "\n")
    except Exception as e:
        print(f"ERROR writing text_boxes.txt: {e}")
