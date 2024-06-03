import cv2
import numpy as np
from skimage.filters import threshold_otsu
from skimage.morphology import closing, square
from skimage.measure import label, regionprops

def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Check if the image was loaded correctly
    if image is None:
        raise FileNotFoundError(f"Image file not found or could not be read: {image_path}")
    # Resize for consistency
    image = cv2.resize(image, (800, 800))
    return image

def edge_detection(image):
    # Apply edge detection
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    return edges

def extract_contours(edges):
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def draw_contours(image, contours):
    # Draw contours on the image
    contour_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
    return contour_image

def threshold_and_segment(image):
    # Thresholding using Otsu's method
    thresh_val = threshold_otsu(image)
    binary = image > thresh_val
    # Closing small holes
    binary = closing(binary, square(3))
    # Label regions
    labeled_image = label(binary)
    return labeled_image

def extract_floor_plan(image_path):
    # Preprocess the image
    image = preprocess_image(image_path)
    
    # Edge detection
    edges = edge_detection(image)
    
    # Extract and draw contours
    contours = extract_contours(edges)
    contour_image = draw_contours(image, contours)
    
    # Threshold and segment
    labeled_image = threshold_and_segment(image)
    
    # Find properties of labeled regions
    regions = regionprops(labeled_image)
    
    # Draw bounding boxes for each region
    floor_plan = np.zeros_like(image)
    for region in regions:
        minr, minc, maxr, maxc = region.bbox
        cv2.rectangle(floor_plan, (minc, minr), (maxc, maxr), (255, 255, 255), 2)
    
    return contour_image, floor_plan

if __name__ == "__main__":
    input_image_path = r'C:\Users\arunv\Downloads\istockphoto-1608315467-1024x1024.jpg'  # Update with your image path
    try:
        contour_image, floor_plan = extract_floor_plan(input_image_path)
        
        # Display the results
        cv2.imshow('Contours', contour_image)
        cv2.imshow('Floor Plan', floor_plan)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except FileNotFoundError as e:
        print(e)
