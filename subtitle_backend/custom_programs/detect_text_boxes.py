"""
Detect text bounding boxes and recognized text in a given video file within a specified time frame using EasyOCR.
Now extended to accept a subtitle file and produce a new ASS file with moved cues if overlap is detected.

USAGE (from command line):
    Basic:  python detect_text_boxes.py --video path/to/video.mp4 --start 10.0 --end 12.0
    Subtitle: python detect_text_boxes.py --video path/to/video.mp4 --subs input.srt --output output.ass

- --video: Path to the video file to analyze.
- --subs:  Subtitle file (.srt or .ass) to analyze for text region overlap
- --output: Output ASS file (required if using --subs)
- --start: Start time in seconds (only for manual mode).
- --end:   End time in seconds (only for manual mode).

OUTPUT:
    Prints and saves (in text_boxes.txt) a list of detected bounding boxes (4 points) and recognized text per box (manual mode)
    For subtitle mode, writes a new ASS file with any cues moved if overlap is detected

Requirements:
    - opencv-python
    - numpy
    - easyocr
    - pysubs2  (for subtitle/ASS handling)
"""
import cv2
import numpy as np
import argparse
import json
import os
import sys
import re
from typing import List, Tuple, Optional

try:
    import easyocr
except ImportError:
    print("ERROR: easyocr is not installed. Run 'pip install easyocr'")
    sys.exit(1)
try:
    import pysubs2
except ImportError:
    print("ERROR: pysubs2 is required for subtitle file handling. Run 'pip install pysubs2'")
    sys.exit(1)

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

def default_subtitle_box(frame_width: int, frame_height: int, region: str = "bottom") -> Tuple[int, int, int, int]:
    """
    Returns (x1, y1, x2, y2) in pixels for the default subtitle region.
    ASS and most players use bottom 18% of video as subtitle region.
    """
    margin_pct = 0.05
    height_ratio = 0.18
    width_margin = int(frame_width * margin_pct)
    height_margin = int(frame_height * margin_pct)
    if region == "bottom":
        x1 = width_margin
        y1 = frame_height - int(height_ratio * frame_height) - height_margin
        x2 = frame_width - width_margin
        y2 = frame_height - height_margin
    else:  # top
        x1 = width_margin
        y1 = height_margin
        x2 = frame_width - width_margin
        y2 = int(height_ratio * frame_height) + height_margin
    return (x1, y1, x2, y2)

def get_frame_indices(start_ms: int, end_ms: int, fps: float, vid_total_frames: int, sample_rate: int) -> List[int]:
    """For a subtitle event [start_ms, end_ms), return a list of frame indices to sample."""
    start_s, end_s = start_ms / 1000.0, end_ms / 1000.0
    indices = []
    t = start_s
    while t < end_s:
        idx = min(int(t * fps), vid_total_frames - 1)
        indices.append(idx)
        t += 1.0 / sample_rate
    return sorted(set(indices))

def text_overlap_in_box(text_results: list, box: Tuple[int, int, int, int]) -> bool:
    """Checks if any OCR-detected text overlaps with the given rectangle."""
    x1, y1, x2, y2 = box
    for result in text_results:
        points = result[0]
        minx = min(p[0] for p in points)
        maxx = max(p[0] for p in points)
        miny = min(p[1] for p in points)
        maxy = max(p[1] for p in points)
        # Overlap if non-disjoint (any intersection)
        if not (maxx < x1 or minx > x2 or maxy < y1 or miny > y2):
            return True
    return False

def process_subtitle_overlap(
    video_path: str,
    subs_path: str,
    output_path: str,
    sample_rate: int = 2,
    ocr_languages: Optional[List[str]] = None,
    gpu: bool = False,
    default_region: str = "bottom"
):
    """
    For each subtitle cue in the subtitle file, uses OCR to check for on-screen text overlap in the default region.
    If overlap is detected, move the subtitle to 'top' (ASS \\an8); otherwise keep region.
    Writes a new ASS file with updated positions.
    """
    ocr_languages = ocr_languages or ["en"]
    # Load subtitles (srt or ass) using pysubs2
    try:
        subs = pysubs2.load(subs_path)
    except Exception as e:
        print("Failed to read subtitles:", e)
        sys.exit(1)
    # Open video and get properties
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Failed to open video file:", video_path)
        sys.exit(2)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video {video_path}: {width}x{height}, {fps} fps, {total_frames} frames")

    reader = easyocr.Reader(ocr_languages, gpu=gpu)
    event_region = {}
    for idx, event in enumerate(subs):
        # Only check Dialogue events; skip comments etc.
        if getattr(event, "type", None) and event.type != "Dialogue":
            continue
        start_ms = event.start
        end_ms = event.end
        frame_indices = get_frame_indices(start_ms, end_ms, fps, total_frames, sample_rate)
        box_default = default_subtitle_box(width, height, default_region)
        overlap_found = False
        # For each sampled frame, check for overlap
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            x1, y1, x2, y2 = box_default
            region_img = frame[y1:y2, x1:x2]
            if region_img.shape[0] < 5 or region_img.shape[1] < 5:
                continue
            # Run EasyOCR on cropped region
            try:
                result = reader.readtext(region_img, detail=1)
            except Exception as ocr_exc:
                print(f"OCR ERROR at event {idx}, frame {frame_idx}: {ocr_exc}")
                continue
            # Offset the bbox back to original image coordinates
            adjusted = []
            for det in result:
                adj_pts = [ (pt[0]+x1, pt[1]+y1) for pt in det[0] ]
                adjusted.append( (adj_pts, det[1], det[2]) )
            if text_overlap_in_box(adjusted, box_default):
                overlap_found = True
                break
        if overlap_found:
            # Switch to the opposite region
            target_region = "top" if default_region == "bottom" else "bottom"
            event_region[event] = target_region
        else:
            event_region[event] = default_region

    # Update subs: Use ASS override tags for region (\an2 = bottom center, \an8 = top center)
    an_map = {"bottom": 2, "top": 8}
    for event in subs:
        if getattr(event, "type", None) and event.type != "Dialogue":
            continue
        region = event_region.get(event, default_region)
        # Remove any explicit \an overrides in-line
        text = event.text
        text_new = re.sub(r"{\\an\\d+}", "", text)
        ass_tag = "{\\an%d}" % an_map[region]
        event.text = ass_tag + text_new

    # Write as ASS file
    try:
        subs.save(output_path, format_="ass")
        print(f"Processed ASS written: {output_path}")
    except Exception as e:
        print("Error saving output ASS:", e)
        sys.exit(3)
    cap.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect text bounding boxes or reposition subtitles if overlap with visible text is found.")
    parser.add_argument("--video", required=True, type=str, help="Path to the video file.")
    parser.add_argument("--fps", default=1, type=int, help="Frames per second to sample (manual mode only, default: 1).")
    parser.add_argument("--lang", type=str, default='en', help="Comma-separated OCR language codes (default: en).")
    parser.add_argument("--gpu", action='store_true', help="Use GPU for OCR")
    parser.add_argument("--subs", type=str, help="Path to subtitle file (SRT or ASS). If specified, manual start/end are ignored and timings are taken from subtitle cues.")
    parser.add_argument("--output", type=str, help="Output ASS file for processed subtitles (required with --subs)")
    parser.add_argument("--sample-rate", type=int, default=2, help="Frames/sec to check for overlap in subtitle mode (default: 2)")
    parser.add_argument("--default-pos", type=str, default="bottom", choices=["bottom", "top"], help="Default subtitle region")
    parser.add_argument("--start", type=float, help="Start time in seconds (manual OCR mode only, ignored if --subs present)")
    parser.add_argument("--end", type=float, help="End time in seconds (manual OCR mode only, ignored if --subs present)")
    args = parser.parse_args()

    language_list = [code.strip() for code in args.lang.split(',') if code.strip()]

    # Subtitle file mode (automatically extracts timings from cue intervals)
    if args.subs:
        if not args.output:
            print("Output ASS file required with --subs.")
            sys.exit(1)
        if not os.path.isfile(args.video):
            print("ERROR: Provided video file does not exist.")
            sys.exit(1)
        if not os.path.isfile(args.subs):
            print("ERROR: Provided subtitle file does not exist.")
            sys.exit(1)
        # All start/stop are extracted inside process_subtitle_overlap from the subtitle cues.
        process_subtitle_overlap(
            args.video, args.subs, args.output,
            sample_rate=args.sample_rate,
            ocr_languages=language_list,
            gpu=args.gpu,
            default_region=args.default_pos,
        )
        sys.exit(0)

    # Manual mode (no subs): require explicit start/end seconds
    if args.subs is None:
        if args.start is None or args.end is None:
            print("For manual OCR mode you must provide --start and --end (not needed if using --subs).")
            sys.exit(1)
        if not os.path.isfile(args.video):
            print("ERROR: Provided video file does not exist.")
            sys.exit(1)
        if args.end < args.start:
            print("ERROR: End time must be after start time.")
            sys.exit(1)
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
