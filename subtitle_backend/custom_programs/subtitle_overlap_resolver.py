#!/usr/bin/env python3
"""
subtitle_overlap_resolver.py

Script to detect on-screen/burnt-in text collisions with subtitle regions in video
and reposition subtitle cues if needed, writing a new .ass file as output.

PUBLIC_INTERFACE
Usage:
    python subtitle_overlap_resolver.py --video path/to/video.mp4 --subs input.srt --output output.ass

Dependencies:
    - easyocr
    - opencv-python
    - numpy
    - pysubs2

Features:
    - Supports SRT or ASS input (output is always ASS)
    - For each subtitle, analyzes frames in its time window
    - Checks for visual text in the default subtitle region using EasyOCR
    - If overlap exists, moves the subtitle higher or to another suitable region
    - Writes new ASS file with updated positions

Author: Kavia subtitle QC project
"""

import argparse
import sys
import cv2
import pysubs2
import easyocr
from typing import Tuple, List

# PUBLIC_INTERFACE
def parse_arguments():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="Subtitle visual overlap detection and repositioning.")
    parser.add_argument("--video", type=str, required=True, help="Input video file (e.g., MP4)")
    parser.add_argument("--subs", type=str, required=True, help="Input subtitle file (SRT/ASS)")
    parser.add_argument("--output", type=str, required=True, help="Output ASS subtitle file")
    parser.add_argument("--sample-rate", type=int, default=2, help="Frames per second to check in each subtitle interval (default: 2)")
    parser.add_argument("--default-pos", type=str, default="bottom", choices=["bottom", "top"], help="Default subtitle display region")
    return parser.parse_args()

# UTILS
def get_frame_indices(start_ms: int, end_ms: int, fps: float, vid_total_frames: int, sample_rate: int) -> List[int]:
    """For a subtitle event [start_ms, end_ms), return a list of frame indices to sample."""
    # Convert ms to seconds
    start_s, end_s = start_ms/1000.0, end_ms/1000.0
    indices = []
    t = start_s
    while t < end_s:
        idx = min(int(t * fps), vid_total_frames-1)
        indices.append(idx)
        t += 1.0/sample_rate
    return sorted(set(indices))

def default_subtitle_box(frame_width: int, frame_height: int, region: str = "bottom") -> Tuple[int, int, int, int]:
    """Returns (x1, y1, x2, y2) in pixels for the default subtitle region."""
    # ASS and most players use bottom 18% of video as subtitle region.
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

def text_overlap_in_box(text_results: list, box: Tuple[int, int, int, int]) -> bool:
    """Checks if any OCR-detected text overlaps with the given rectangle."""
    x1, y1, x2, y2 = box
    for result in text_results:
        # result[0] is four points: [(x1,y1),...(x4,y4)] covering the text box.
        points = result[0]
        # Get bounding box of detected text region
        minx = min(p[0] for p in points)
        maxx = max(p[0] for p in points)
        miny = min(p[1] for p in points)
        maxy = max(p[1] for p in points)
        # Simple overlap check: if boxes intersect (any overlap at all)
        if not (maxx < x1 or minx > x2 or maxy < y1 or miny > y2):
            return True
    return False

# PUBLIC_INTERFACE
def main():
    """Main script entry point."""
    args = parse_arguments()
    video_path = args.video
    subs_path = args.subs
    output_path = args.output
    sample_rate = args.sample_rate
    default_region = args.default_pos

    # Load subtitles (SRT or ASS accepted, always output as ASS)
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
    
    reader = easyocr.Reader(['en'])  # could parameterize for language

    # Prepare map of new positions for events {event: region_name}
    event_region = {}

    for idx, event in enumerate(subs):
        # Only Dialogue events; pysubs2 SRT loads into SSAEvent with type DIALOGUE
        if getattr(event, "type", None) and event.type != "Dialogue":
            continue
        start_ms = event.start
        end_ms = event.end
        frame_indices = get_frame_indices(start_ms, end_ms, fps, total_frames, sample_rate)
        box_default = default_subtitle_box(width, height, default_region)
        overlap_found = False

        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                print(f"Warning: could not read frame {frame_idx}")
                continue
            # Crop region for minimal computation
            x1, y1, x2, y2 = box_default
            region_img = frame[y1:y2, x1:x2]
            if region_img.shape[0] < 5 or region_img.shape[1] < 5:
                continue
            # Run EasyOCR
            result = reader.readtext(region_img, detail=1)
            # Map text boxes back to video coordinates for the overlap test
            # Apply offset to OCR bounding boxes:
            adjusted = []
            for det in result:
                adj_pts = [ (pt[0]+x1, pt[1]+y1) for pt in det[0] ]
                adjusted.append( (adj_pts, det[1], det[2]) )
            if text_overlap_in_box(adjusted, box_default):
                overlap_found = True
                break

        if overlap_found:
            # Try the other region: move to top if default is bottom (and vice versa)
            target_region = "top" if default_region == "bottom" else "bottom"
            # Optional: You could run OCR on the alternate region to check for conflicts too
            event_region[event] = target_region
        else:
            event_region[event] = default_region

    # Update subs: ASS supports override tags for position (\pos, \move, or \an alignment).
    # We'll use \an2 (bottom center; default ASS) and \an8 (top center).
    an_map = {"bottom": 2, "top": 8}
    for event in subs:
        if getattr(event, "type", None) and event.type != "Dialogue":
            continue
        region = event_region.get(event, default_region)
        # Remove existing \an tags in override (if any)
        text = event.text
        import re
        text_new = re.sub(r"{\\an\d+}", "", text)
        ass_tag = "{\\an%d}" % an_map[region]
        event.text = ass_tag + text_new

    # Save as ASS file
    try:
        subs.save(output_path, format_="ass")
        print(f"Processed and saved as ASS: {output_path}")
    except Exception as e:
        print("Error saving:", e)
        sys.exit(3)
    cap.release()

if __name__ == "__main__":
    main()
