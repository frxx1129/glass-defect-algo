"""
Exclusion Zone Annotator

A tool to annotate exclusion zones (non-detection areas) on images.
The output can be directly copied into the code.

Usage:
1. Run the script and select the image to annotate (line2cam3.BMP or line3cam3.BMP)
2. Hold left mouse button and drag to draw rectangular exclusion zones (green rectangles)
3. Press R to reset all zones
4. Press D to delete the last zone
5. Press S to save and output Python code format
6. Press Q to quit

Author: LI Zhaoyang
Date: 2026-01-05
"""

import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog


class ExclusionZoneAnnotator:
    def __init__(self):
        self.ref_point_start = None
        self.ref_point_end = None
        self.drawing = False
        self.exclusion_zones = []
        self.image_clone = None
        self.original_image = None
        self.scale_factor = 1.0
        self.MAX_DISPLAY_WIDTH = 2000
        self.MAX_DISPLAY_HEIGHT = 1200
        self.image_path = None

    def display_help_text(self, image):
        """Display help text on the image"""
        help_lines = [
            "Instructions:",
            "  Left mouse drag: Draw rectangle",
            "  R: Reset all zones",
            "  D: Delete last zone",
            "  S: Save and output code",
            "  Q: Quit"
        ]
        y_offset = 25
        for line in help_lines:
            cv2.putText(image, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (0, 0, 255), 1, cv2.LINE_AA)
            y_offset += 22
        
        # Display current zone count
        count_text = f"Zones marked: {len(self.exclusion_zones)}"
        cv2.putText(image, count_text, (10, y_offset + 10), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, (0, 255, 0), 2, cv2.LINE_AA)
        return image

    def draw_zones(self, image, zones):
        """Draw all exclusion zones"""
        for i, zone in enumerate(zones):
            x, y, w, h = zone['x'], zone['y'], zone['width'], zone['height']
            # Draw filled area (semi-transparent green)
            overlay = image.copy()
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), -1)
            cv2.addWeighted(overlay, 0.2, image, 0.8, 0, image)
            # Draw border (thin line for precise annotation)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
            # Draw zone number
            cv2.putText(image, f"Zone {i+1}", (x + 5, y + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            # Display zone size
            size_text = f"{w}x{h}"
            cv2.putText(image, size_text, (x + 5, y + h - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        return image

    def mouse_callback(self, event, x, y, flags, param):
        """Mouse event callback"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.ref_point_start = (x, y)
            self.drawing = True
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.ref_point_end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.ref_point_end = (x, y)
            self.drawing = False
            if self.ref_point_start and self.ref_point_end:
                x1, y1 = self.ref_point_start
                x2, y2 = self.ref_point_end
                # Ensure rectangle is valid (width and height > 3 pixels for fine annotation)
                if abs(x2 - x1) > 3 and abs(y2 - y1) > 3:
                    start_x, start_y = min(x1, x2), min(y1, y2)
                    end_x, end_y = max(x1, x2), max(y1, y2)
                    # Convert to original image coordinates
                    original_x = int(start_x * self.scale_factor)
                    original_y = int(start_y * self.scale_factor)
                    original_w = int((end_x - start_x) * self.scale_factor)
                    original_h = int((end_y - start_y) * self.scale_factor)
                    self.exclusion_zones.append({
                        'x': original_x,
                        'y': original_y,
                        'width': original_w,
                        'height': original_h
                    })
                    print(f"Added zone {len(self.exclusion_zones)}: x={original_x}, y={original_y}, "
                          f"width={original_w}, height={original_h}")
            self.ref_point_start = None
            self.ref_point_end = None

    def generate_code_output(self):
        """Generate Python code that can be copied into the code"""
        if not self.exclusion_zones:
            print("\nNo zones marked!")
            return ""
        
        # Determine Line2 or Line3 based on filename
        filename = os.path.basename(self.image_path).lower()
        if 'line2' in filename:
            var_name = "LINE2_CAM3_EXCLUSION_ZONES"
            comment = "# Line2 cam3 exclusion zones"
        elif 'line3' in filename:
            var_name = "LINE3_CAM3_EXCLUSION_ZONES"
            comment = "# Line3 cam3 exclusion zones"
        else:
            var_name = "EXCLUSION_ZONES"
            comment = "# Exclusion zones"
        
        code_lines = [
            "",
            "=" * 70,
            comment,
            f"{var_name} = [",
        ]
        
        for i, zone in enumerate(self.exclusion_zones):
            comma = "," if i < len(self.exclusion_zones) - 1 else ""
            code_lines.append(f'    {{"x": {zone["x"]}, "y": {zone["y"]}, '
                            f'"width": {zone["width"]}, "height": {zone["height"]}}}{comma}')
        
        code_lines.append("]")
        code_lines.append("=" * 70)
        code_lines.append("")
        
        code_str = "\n".join(code_lines)
        print(code_str)
        
        # Also save to file
        output_filename = os.path.splitext(self.image_path)[0] + "_exclusion_zones.py"
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(f'"""\nAuto-generated exclusion zone definition\nSource image: {os.path.basename(self.image_path)}\n"""\n\n')
            f.write(f"{comment}\n")
            f.write(f"{var_name} = [\n")
            for i, zone in enumerate(self.exclusion_zones):
                comma = "," if i < len(self.exclusion_zones) - 1 else ""
                f.write(f'    {{"x": {zone["x"]}, "y": {zone["y"]}, '
                       f'"width": {zone["width"]}, "height": {zone["height"]}}}{comma}\n')
            f.write("]\n")
        print(f"Zone definition saved to: {output_filename}")
        
        return code_str

    def run_annotator(self, image_path):
        """Run the annotator"""
        self.image_path = image_path
        
        if not os.path.isfile(image_path):
            print(f"File not found: {image_path}")
            return None
        
        # Read image
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            print(f"Unable to read image: {image_path}")
            return None
        
        original_h, original_w = self.original_image.shape[:2]
        print(f"Original image size: {original_w} x {original_h}")
        
        # Calculate scale factor
        scale_w = self.MAX_DISPLAY_WIDTH / original_w if original_w > self.MAX_DISPLAY_WIDTH else 1.0
        scale_h = self.MAX_DISPLAY_HEIGHT / original_h if original_h > self.MAX_DISPLAY_HEIGHT else 1.0
        self.scale_factor = max(1.0, 1.0 / min(scale_w, scale_h))
        
        if self.scale_factor > 1.0:
            display_w = int(original_w / self.scale_factor)
            display_h = int(original_h / self.scale_factor)
            display_image = cv2.resize(self.original_image, (display_w, display_h), 
                                      interpolation=cv2.INTER_AREA)
            print(f"Display size: {display_w} x {display_h} (scale: 1:{self.scale_factor:.2f})")
        else:
            self.scale_factor = 1.0
            display_image = self.original_image.copy()
        
        self.image_clone = display_image.copy()
        
        # Create window
        window_name = f"Exclusion Zone Annotator - {os.path.basename(image_path)}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        print("\n--- Exclusion Zone Annotator ---")
        print("Instructions:")
        print(" - Left mouse button drag to draw rectangular exclusion zones")
        print(" - R: Reset all zones")
        print(" - D: Delete last zone")
        print(" - S: Save and output code")
        print(" - Q: Quit")
        print("-" * 30)
        
        while True:
            # Copy original display image
            current_display = self.image_clone.copy()
            
            # Draw saved zones (convert to display coordinates)
            display_zones = []
            for zone in self.exclusion_zones:
                display_zones.append({
                    'x': int(zone['x'] / self.scale_factor),
                    'y': int(zone['y'] / self.scale_factor),
                    'width': int(zone['width'] / self.scale_factor),
                    'height': int(zone['height'] / self.scale_factor)
                })
            current_display = self.draw_zones(current_display, display_zones)
            
            # Draw rectangle being dragged
            if self.drawing and self.ref_point_start and self.ref_point_end:
                cv2.rectangle(current_display, self.ref_point_start, self.ref_point_end, 
                            (0, 255, 255), 1)
            
            # Display help text
            current_display = self.display_help_text(current_display)
            
            cv2.imshow(window_name, current_display)
            
            key = cv2.waitKey(10) & 0xFF
            
            if key in [ord('r'), ord('R')]:
                self.exclusion_zones = []
                print("All zones reset")
            elif key in [ord('d'), ord('D')]:
                if self.exclusion_zones:
                    removed = self.exclusion_zones.pop()
                    print(f"Deleted zone: x={removed['x']}, y={removed['y']}")
                else:
                    print("No zones to delete")
            elif key in [ord('s'), ord('S')]:
                self.generate_code_output()
            elif key in [ord('q'), ord('Q')]:
                break
        
        cv2.destroyAllWindows()
        return self.exclusion_zones


def main():
    print("=" * 50)
    print("Exclusion Zone Annotator")
    print("=" * 50)
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check for default image files
    default_images = [
        os.path.join(script_dir, "line2cam3.BMP"),
        os.path.join(script_dir, "Line2cam3.BMP"),
        os.path.join(script_dir, "line3cam3.BMP"),
        os.path.join(script_dir, "Line3cam3.BMP"),
    ]
    
    available_images = [img for img in default_images if os.path.exists(img)]
    
    if available_images:
        print("\nDetected images available for annotation:")
        for i, img in enumerate(available_images):
            print(f"  {i + 1}. {os.path.basename(img)}")
        print(f"  {len(available_images) + 1}. Select other file...")
        
        while True:
            try:
                choice = input(f"\nPlease select (1-{len(available_images) + 1}): ").strip()
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(available_images):
                    image_path = available_images[choice_idx]
                    break
                elif choice_idx == len(available_images):
                    # Select other file
                    root = tk.Tk()
                    root.withdraw()
                    image_path = filedialog.askopenfilename(
                        title="Select image to annotate",
                        initialdir=script_dir,
                        filetypes=[
                            ("BMP files", "*.bmp"),
                            ("All images", "*.bmp;*.jpg;*.jpeg;*.png"),
                            ("All files", "*.*")
                        ]
                    )
                    if not image_path:
                        print("No file selected, exiting.")
                        return
                    break
                else:
                    print("Invalid choice, please try again")
            except ValueError:
                print("Please enter a valid number")
    else:
        # No default images, open file dialog
        root = tk.Tk()
        root.withdraw()
        image_path = filedialog.askopenfilename(
            title="Select image to annotate",
            initialdir=script_dir,
            filetypes=[
                ("BMP files", "*.bmp"),
                ("All images", "*.bmp;*.jpg;*.jpeg;*.png"),
                ("All files", "*.*")
            ]
        )
        if not image_path:
            print("No file selected, exiting.")
            return
    
    print(f"\nOpening: {image_path}")
    
    annotator = ExclusionZoneAnnotator()
    zones = annotator.run_annotator(image_path)
    
    if zones:
        print(f"\nAnnotation complete, {len(zones)} exclusion zone(s)")
        annotator.generate_code_output()
    else:
        print("\nNo zones marked")


if __name__ == '__main__':
    main()
