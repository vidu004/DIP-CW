import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import matplotlib.pyplot as plt
import os

# Global variables for storing image and analysis results
image_data = []



# Maximum display size for images in the GUI
MAX_DISPLAY_WIDTH = 500
MAX_DISPLAY_HEIGHT = 400

# Function to check if the image contains skin tones
def validate_image_for_skin(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define HSV range for detecting skin tones
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Create a mask for skin tone areas
    skin_mask = cv2.inRange(hsv_image, lower_skin, upper_skin)

    # Check if the skin tone area is significant enough
    skin_area = cv2.countNonZero(skin_mask)
    total_area = image.shape[0] * image.shape[1]
    skin_ratio = skin_area / total_area

    # If skin area covers more than 5% of the image, we assume it contains human skin
    return skin_ratio > 0.05

# Modify the upload_image function to validate each image for skin tones
def upload_image():
    file_paths = filedialog.askopenfilenames(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
    if not file_paths:
        return

    for file_path in file_paths:
        if any(entry["file_path"] == file_path for entry in image_data):
            messagebox.showwarning("Duplicate Image", f"The image '{os.path.basename(file_path)}' is already uploaded.")
            continue

        img = cv2.imread(file_path)
        if img is None:
            messagebox.showerror("Error", f"Failed to load image: {file_path}")
            continue

        # Check if the image contains skin tones
        if not validate_image_for_skin(img):
            messagebox.showwarning("No Skin Detected", f"No skin lesion detected in '{os.path.basename(file_path)}'. Skipping.")
            continue

        processed_img = preprocess_image(img)
        mask = detect_skin_changes(img)
        processed_img, lesion_info, lesion_spread = analyze_changes(mask, img.copy())
        condition, roughness, lesion_ages = analyze_skin_condition(lesion_spread, lesion_info)

        image_entry = {
            "file_path": file_path,
            "original_image": img,
            "processed_image": processed_img,
            "lesion_info": lesion_info,
            "lesion_spread": lesion_spread,
            "condition": condition,
            "roughness": roughness,
            "lesion_ages": lesion_ages
        }
        image_data.append(image_entry)

    if image_data:
        listbox.delete(0, tk.END)
        for idx, entry in enumerate(image_data):
            file_name = os.path.basename(entry["file_path"])
            listbox.insert(tk.END, f"{idx + 1}. {file_name}")
        
        listbox.select_set(0)
        on_select(None)

        update_status(f"{len(image_data)} image(s) uploaded and processed successfully.")

# Function to display image in the GUI
def display_image(image, title):
    if image is None:
        return
    
    height, width = image.shape[:2]
    size_text = f"Dimensions: {width} x {height} pixels"
    
    display_img = resize_image(image, MAX_DISPLAY_WIDTH, MAX_DISPLAY_HEIGHT)
    
    image_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    
    try:
        image_tk = ImageTk.PhotoImage(image_pil)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to create image: {e}")
        return

    image_label.config(image=image_tk)
    image_label.image = image_tk
    image_label_text.set(title)
    size_label.config(text=size_text)

# Function to resize images
def resize_image(image, max_width, max_height):
    height, width = image.shape[:2]
    aspect_ratio = width / height

    if width > max_width or height > max_height:
        if aspect_ratio > 1:
            new_width = max_width
            new_height = int(max_width / aspect_ratio)
        else:
            new_height = max_height
            new_width = int(max_height * aspect_ratio)
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return resized_image
    return image

# Preprocess the image for analysis
def preprocess_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (7, 7), 0)
    return blurred_image

def detect_skin_changes(image):
    # Convert to LAB color space to isolate brightness (L channel)
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel to enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l_channel = clahe.apply(l_channel)
    lab_image = cv2.merge((enhanced_l_channel, a_channel, b_channel))
    enhanced_image = cv2.cvtColor(lab_image, cv2.COLOR_Lab2BGR)
    
    # Use adaptive thresholding for better lesion area detection
    gray_image = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)
    _, lesion_mask = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Apply Canny edge detection on enhanced L channel
    edges = cv2.Canny(enhanced_l_channel, 30, 100)
    
    # Refine the lesion mask by combining with edges
    combined_mask = cv2.bitwise_and(lesion_mask, edges)
    
    # Morphological operations to refine the edges and remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Find contours and draw only significant contours for clearer lesions
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 100:  # Filter small contours by area
            cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)

    return combined_mask

# Function to analyze changes in the image
def analyze_changes(mask, original_image):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_skin_area = original_image.shape[0] * original_image.shape[1]
    total_lesion_area = 0
    lesion_info = []
    
    PIXELS_PER_MM = 10  # Change this to your actual pixel density
    PIXELS_PER_MM_SQ = PIXELS_PER_MM ** 2

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 1500:
            continue

        total_lesion_area += area
        hull = cv2.convexHull(contour)
        cv2.drawContours(original_image, [hull], -1, (0, 255, 0), 2)
        perimeter = cv2.arcLength(contour, True)
        hull_area = cv2.contourArea(hull)
        convexity_ratio = area / hull_area if hull_area != 0 else 0
        mask_contour = np.zeros(original_image.shape[:2], np.uint8)
        cv2.drawContours(mask_contour, [contour], -1, 255, -1)
        mean_val = cv2.mean(original_image, mask=mask_contour)
        lesion_area_mm = area / PIXELS_PER_MM_SQ
        perimeter_mm = perimeter / PIXELS_PER_MM

        lesion_info.append({
            "area": lesion_area_mm,
            "perimeter": perimeter_mm,
            "convexity_ratio": convexity_ratio,
            "mean_color": mean_val
        })

    total_skin_area_mm = total_skin_area / PIXELS_PER_MM_SQ
    total_lesion_area_mm = total_lesion_area / PIXELS_PER_MM_SQ
    lesion_spread = (total_lesion_area_mm / total_skin_area_mm) * 100

    return original_image, lesion_info, lesion_spread

# Function to analyze skin condition based on lesions
def analyze_skin_condition(lesion_spread, lesion_info):
    if lesion_spread < 10:
        condition = "Mild Lesion Spread"
    elif 10 <= lesion_spread <= 20:
        condition = "Moderate Lesion Spread"
    else:
        condition = "Severe Lesion Spread"

    for lesion in lesion_info:
        if lesion['convexity_ratio'] < 0.85:
            condition += " with Irregular Shape (Potential Risk)"
            break

    roughness = calculate_skin_roughness(lesion_info)

    lesion_ages = []
    for lesion in lesion_info:
        lesion_ages.append(estimate_lesion_age(lesion))

    return condition, roughness, lesion_ages

# Function to calculate skin roughness based on lesions
def calculate_skin_roughness(lesion_info):
    roughness_scores = []
    for lesion in lesion_info:
        mean_color = lesion['mean_color']
        roughness_score = np.std(mean_color)
        roughness_scores.append(roughness_score)
    
    if roughness_scores:
        avg_roughness = np.mean(roughness_scores)
    else:
        avg_roughness = 0

    if avg_roughness < 25:
        return "Smooth Skin"
    elif 25 <= avg_roughness < 50:
        return "Moderately Rough Skin"
    else:
        return "Rough Skin"

# Function to estimate lesion age based on color
def estimate_lesion_age(lesion):
    mean_color = lesion['mean_color']
    red, green, blue = mean_color[:3]
    
    if red > green and red > blue:
        return "Recent Lesion (1-2 days)"
    elif green > red and green > blue:
        return "Moderately Old Lesion (3-7 days)"
    elif blue > red and blue > green:
        return "Older Lesion (7+ days)"
    else:
        return "Unclassified Lesion Age"

# Function to handle image selection in the listbox
def on_select(event):
    selection = listbox.curselection()
    if not selection:
        return
    index = selection[0]
    selected_image_entry = image_data[index]
    display_image(selected_image_entry["original_image"], "Original Image")

# Function to update the status bar
def update_status(message):
    status_label.config(text=message) # type: ignore

# Function to export the analysis results to a PDF
def export_results_as_pdf():
    if not image_data:
        messagebox.showwarning("No Data", "Please upload and process at least one image before exporting.")
        return

    file_path = filedialog.asksaveasfilename(defaultextension=".pdf", 
                                             filetypes=[("PDF Files", "*.pdf")],
                                             title="Save Results As")
    if not file_path:
        return

    try:
        c = canvas.Canvas(file_path, pagesize=letter)
        width, height = letter

        for idx, entry in enumerate(image_data):
            # Draw the image
            image_pil = Image.fromarray(cv2.cvtColor(entry["original_image"], cv2.COLOR_BGR2RGB))
            temp_image_path = f"temp_image_{idx}.png"
            image_pil.save(temp_image_path)
            c.drawImage(temp_image_path, 50, height - 300, width=500, height=250)
            os.remove(temp_image_path)

            # Set font and starting Y-coordinate for text
            c.setFont("Helvetica", 10)
            y_position = height - 320  # Start position for text below the image

            # Draw general information
            c.drawString(50, y_position, f"Image {idx + 1}: {os.path.basename(entry['file_path'])}")
            y_position -= 20
            c.drawString(50, y_position, f"Skin Condition: {entry['condition']}")
            y_position -= 20
            c.drawString(50, y_position, f"Skin Roughness: {entry['roughness']}")
            y_position -= 20
            c.drawString(50, y_position, f"Lesion Spread: {entry['lesion_spread']:.2f}%")
            y_position -= 30  # Add extra spacing before lesion details

            # Draw lesion information
            for i, lesion in enumerate(entry["lesion_info"]):
                c.drawString(50, y_position, 
                             f"Lesion {i + 1}: Area: {lesion['area']:.2f} mm², Perimeter: {lesion['perimeter']:.2f} mm, Convexity: {lesion['convexity_ratio']:.2f}")
                y_position -= 20
                c.drawString(50, y_position, 
                             f"Mean Color (BGR): {int(lesion['mean_color'][0])}, {int(lesion['mean_color'][1])}, {int(lesion['mean_color'][2])}")
                y_position -= 20
                c.drawString(50, y_position, f"Estimated Age: {entry['lesion_ages'][i]}")
                y_position -= 30  # Add extra space after each lesion

                # Check if there is enough space left on the page, if not, add a new page
                if y_position < 100:
                    c.showPage()
                    y_position = height - 50  # Reset Y position on the new page

            c.showPage()

        c.save()
        messagebox.showinfo("Export Successful", f"Results have been successfully exported to {file_path}")
    except Exception as e:
        messagebox.showerror("Export Error", f"Failed to export PDF: {e}")
        

def display_histogram_with_lesion(image, lesion_mask=None):
    if image is None:
        return

    colors = ('b', 'g', 'r')
    channel_names = {'b': 'Blue', 'g': 'Green', 'r': 'Red'}

    plt.figure(figsize=(12, 8))
    plt.title('Color Distribution and Lesion Areas', fontsize=14)
    plt.xlabel('Color Shade', fontsize=12)
    plt.ylabel('Count of Pixels in Lesions', fontsize=12)
    plt.xlim([0, 256])
    plt.grid(True, linestyle='--', alpha=0.7)

    # Calculate and plot histograms for each color channel
    for i, col in enumerate(colors):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=col, label=f'{channel_names[col]} Color')

    # Overlay lesion information if provided
    max_lesion_count = 0  # Initialize variable
    if lesion_mask is not None:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lesion_hist = cv2.calcHist([gray_image], [0], lesion_mask, [256], [0, 256])
        plt.plot(lesion_hist, color='orange', label='Lesion Area', alpha=0.7)

        # Find max lesion count for annotation
        max_lesion_count = np.max(lesion_hist)

        # Add vertical lines to indicate where lesions are detected
        lesion_indices = np.where(lesion_hist > 0)[0]
        for idx in lesion_indices:
            plt.axvline(x=idx, color='orange', linestyle='--', alpha=0.5)

        # Annotate lesion spread
        if len(lesion_indices) > 0:  # Check if any lesions were found
            plt.annotate('Lesion Area', xy=(lesion_indices[0], max_lesion_count), 
                         xytext=(lesion_indices[0] + 10, max_lesion_count - 50),
                         arrowprops=dict(facecolor='orange', shrink=0.05),
                         fontsize=10, color='orange')

    # Legend for color channels
    plt.legend(loc='upper right', fontsize=10)

    # Add informative text outside the graph area
    plt.text(0.5, -0.1, 'Higher peaks indicate more lesions in those color shades.', 
             ha='center', fontsize=12, color='black', wrap=True, transform=plt.gca().transAxes)

    plt.tight_layout(rect=[0, 0.1, 1, 1])  # Adjust layout to make room for text
    plt.show()


# Function to handle histogram display button
def show_histogram():
    selection = listbox.curselection()  # Assuming `listbox` is a Tkinter Listbox widget
    if not selection:
        messagebox.showwarning("No Selection", "Please select an image first.")
        return
    
    index = selection[0]
    selected_image_entry = image_data[index]  # Assuming `image_data` is a list of images
    
    # Load the original image and lesion mask from the selected entry
    original_image = selected_image_entry["original_image"]
    lesion_mask = selected_image_entry.get("lesion_mask")  # Ensure lesion mask is available

    # Call the histogram display function
    display_histogram_with_lesion(original_image, lesion_mask)


   
    # Function to remove the selected uploaded image
def remove_image():
    selection = listbox.curselection()
    if not selection:
        messagebox.showwarning("No Selection", "Please select an image to remove.")
        return

    index = selection[0]
    removed_image = image_data.pop(index)

    listbox.delete(index)

    if image_data:
        listbox.select_set(0)
        on_select(None)
        update_status(f"Removed: {os.path.basename(removed_image['file_path'])}. {len(image_data)} image(s) remaining.")
    else:
        image_label.config(image='')
        image_label_text.set("No image selected")
        size_label.config(text="")
        update_status("All images removed. No images to display.")

# Function to update status bar when no images are selected
def update_status(message):
    status_label.config(text=message)


# GUI setup
root = tk.Tk()
root.title("Skin Lesion Analysis Tool")
root.geometry("800x600")
root.configure(bg="#d4f1f4")  # Light blue background

# Image Display
image_label = tk.Label(root, bg="#d4f1f4")
image_label.grid(row=0, column=0, padx=10, pady=10)

# Image Text
image_label_text = tk.StringVar()
image_label_text.set("No Image Loaded")
tk.Label(root, textvariable=image_label_text, bg="#d4f1f4", font=("Arial", 12)).grid(row=1, column=0)

# Size label
size_label = tk.Label(root, text="Dimensions: ", bg="#d4f1f4", font=("Arial", 12))
size_label.grid(row=2, column=0)

# Listbox to show image files
listbox = tk.Listbox(root, height=15, bg="#ffffff", font=("Arial", 12))
listbox.grid(row=0, column=1, rowspan=3, padx=10, pady=10, sticky="ns")
listbox.bind("<<ListboxSelect>>", on_select)

# Buttons
btn_upload = tk.Button(root, text="Upload Image(s)", command=upload_image, bg="#4caf50", fg="white", font=("Arial", 12))
btn_upload.grid(row=3, column=0, padx=5, pady=5, sticky="ew")

btn_export = tk.Button(root, text="Export as PDF", command=export_results_as_pdf, bg="#2196f3", fg="white", font=("Arial", 12))
btn_export.grid(row=3, column=1, padx=5, pady=5, sticky="ew")

btn_histogram = tk.Button(root, text="Show Histogram", command=show_histogram, bg="#ff9800", fg="white", font=("Arial", 12))
btn_histogram.grid(row=4, column=1, columnspan=2, padx=5, pady=5, sticky="ew")

# Make buttons responsive
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)

# GUI Button for removing the selected image
remove_button = tk.Button(root, text="Remove Image", command=remove_image, bg="#f44336", fg="white", font=("Arial", 12))
remove_button.grid(row=4, column=0, padx=5, pady=5, sticky="ew")



# Make buttons responsive
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)

# Run the application
root.mainloop()