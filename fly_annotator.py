import cv2
import os
import json
import glob
import numpy as np
import tkinter as tk
from tkinter import filedialog

class FlyAnnotator:
    def __init__(self, image_folder="flyings", output_file="fly_features.json"):
        self.image_folder = image_folder
        self.output_file = output_file
        self.image_files = []
        self.current_img_idx = 0
        
        self.original_image = None
        self.display_image = None
        self.scale_factor = 1.0
        
        self.drawing = False
        self.start_pt = None
        self.end_pt = None
        
        self.current_boxes = []
        self.all_features = []
        
        self.load_images()

    def load_images(self):
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        for ext in extensions:
            self.image_files.extend(glob.glob(os.path.join(self.image_folder, ext)))
        self.image_files = sorted(self.image_files)
        print(f"Found {len(self.image_files)} images in {self.image_folder}.")

    def calculate_features(self, img, box):
        x, y, w, h = box
        # Extract the crop safely
        crop = img[y:y+h, x:x+w]
        if crop.size == 0:
            return None
        
        # Convert to grayscale
        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop
        
        # Calculate features
        area = w * h
        aspect_ratio = w / h if h > 0 else 0
        
        # Mean intensity
        mean_intensity = np.mean(gray)
        
        # Standard deviation (contrast)
        std_dev = np.std(gray)
        
        # Blur/Focus measure (Variance of Laplacian)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Try to find the bug contour to calculate solidity
        _, thresh = cv2.threshold(gray, mean_intensity * 0.9, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        solidity = 0.0
        contour_area = 0.0
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            contour_area = cv2.contourArea(largest_contour)
            hull = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = contour_area / hull_area
                
        extent = contour_area / area if area > 0 else 0
        
        return {
            "bbox_w": w,
            "bbox_h": h,
            "bbox_area": area,
            "aspect_ratio": round(aspect_ratio, 3),
            "mean_intensity": round(mean_intensity, 3),
            "std_dev": round(std_dev, 3),
            "laplacian_var": round(laplacian_var, 3),
            "solidity": round(solidity, 3),
            "extent": round(extent, 3)
        }

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_pt = (x, y)
            self.end_pt = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.end_pt = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end_pt = (x, y)
            
            x1, y1 = self.start_pt
            x2, y2 = self.end_pt
            
            # Ensure proper coordinates and minimum size
            left, right = min(x1, x2), max(x1, x2)
            top, bottom = min(y1, y2), max(y1, y2)
            
            w = right - left
            h = bottom - top
            
            if w > 5 and h > 5:
                # Map back to original image coordinates
                orig_x = int(left / self.scale_factor)
                orig_y = int(top / self.scale_factor)
                orig_w = int(w / self.scale_factor)
                orig_h = int(h / self.scale_factor)
                
                self.current_boxes.append((orig_x, orig_y, orig_w, orig_h))
                print(f"Added box: {orig_w}x{orig_h} at ({orig_x},{orig_y})")
                
            self.start_pt = None
            self.end_pt = None

    def display_current_image(self):
        if self.current_img_idx >= len(self.image_files):
            print("All images processed!")
            self.save_features()
            return False
            
        img_path = self.image_files[self.current_img_idx]
        self.original_image = cv2.imread(img_path)
        if self.original_image is None:
            print(f"Failed to load {img_path}")
            self.current_img_idx += 1
            return True
            
        # Calculate display scale
        screen_w, screen_h = 1600, 900
        orig_h, orig_w = self.original_image.shape[:2]
        
        scale_w = screen_w / orig_w
        scale_h = screen_h / orig_h
        self.scale_factor = min(1.0, scale_w, scale_h)
        
        display_w = int(orig_w * self.scale_factor)
        display_h = int(orig_h * self.scale_factor)
        
        self.display_image = cv2.resize(self.original_image, (display_w, display_h))
        self.current_boxes = []
        
        window_name = "Fly Annotator"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        while True:
            img_copy = self.display_image.copy()
            
            # Draw existing boxes
            for i, (ox, oy, ow, oh) in enumerate(self.current_boxes):
                x = int(ox * self.scale_factor)
                y = int(oy * self.scale_factor)
                w = int(ow * self.scale_factor)
                h = int(oh * self.scale_factor)
                cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img_copy, f"Fly {i+1}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
            # Draw current drag box
            if self.drawing and self.start_pt and self.end_pt:
                cv2.rectangle(img_copy, self.start_pt, self.end_pt, (0, 0, 255), 2)
                
            # Info text
            text = f"Image {self.current_img_idx+1}/{len(self.image_files)} | File: {os.path.basename(img_path)}"
            cv2.putText(img_copy, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(img_copy, "Left Click + Drag: Draw | D: Delete Last | Space: Next Image | S: Save&Quit | Q: Quit", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            cv2.imshow(window_name, img_copy)
            key = cv2.waitKey(20) & 0xFF
            
            if key == ord('d') or key == ord('D'):
                if self.current_boxes:
                    self.current_boxes.pop()
                    print("Deleted last box.")
            elif key == ord(' '): # Space
                # Save features for current boxes
                for box in self.current_boxes:
                    features = self.calculate_features(self.original_image, box)
                    if features:
                        features["image_file"] = os.path.basename(img_path)
                        self.all_features.append(features)
                print(f"Saved {len(self.current_boxes)} flies from current image.")
                self.current_img_idx += 1
                break
            elif key == ord('s') or key == ord('S'):
                # Save current and quit
                for box in self.current_boxes:
                    features = self.calculate_features(self.original_image, box)
                    if features:
                        features["image_file"] = os.path.basename(img_path)
                        self.all_features.append(features)
                self.save_features()
                return False
            elif key == ord('q') or key == ord('Q'):
                print("Quit without saving latest changes.")
                return False
                
        return True

    def save_features(self):
        if not self.all_features:
            print("No features to save.")
            return
            
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(self.all_features, f, indent=4)
        print(f"Successfully saved {len(self.all_features)} insect features to {self.output_file}")
        
        # Print summary statistics
        areas = [f["bbox_area"] for f in self.all_features]
        aspect_ratios = [f["aspect_ratio"] for f in self.all_features]
        solidities = [f["solidity"] for f in self.all_features]
        
        print("\n--- Feature Summary ---")
        print(f"Area (px): Min={min(areas)}, Max={max(areas)}, Avg={np.mean(areas):.1f}")
        print(f"Aspect Ratio: Min={min(aspect_ratios)}, Max={max(aspect_ratios)}, Avg={np.mean(aspect_ratios):.2f}")
        print(f"Solidity: Min={min(solidities)}, Max={max(solidities)}, Avg={np.mean(solidities):.2f}")

    def run(self):
        if not self.image_files:
            return
        print("Starting annotator...")
        while self.display_current_image():
            pass
        cv2.destroyAllWindows()

if __name__ == "__main__":
    annotator = FlyAnnotator()
    annotator.run()
