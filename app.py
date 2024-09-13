from flask import Flask, request, render_template, redirect, url_for
import cv2
import dlib
import os
from PIL import Image, ImageFilter

app = Flask(__name__)

# Load pre-trained face detector and facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def get_landmarks(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image file '{image_path}' not found or unable to open.")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    landmarks = []
    for face in faces:
        shape = predictor(gray, face)
        landmarks.append([(p.x, p.y) for p in shape.parts()])
    return image, landmarks

def draw_landmarks(image, landmarks):
    for landmark in landmarks:
        for (x, y) in landmark:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
    return image

def apply_filter(image_path, filter_type):
    image = Image.open(image_path)
    if filter_type == 'BLUR':
        image = image.filter(ImageFilter.BLUR)
    elif filter_type == 'CONTOUR':
        image = image.filter(ImageFilter.CONTOUR)
    elif filter_type == 'DETAIL':
        image = image.filter(ImageFilter.DETAIL)
    # Add more filters if needed
    return image

@app.route('/', methods=['GET', 'POST'])
def index():
    error_message = None  # Initialize error message variable
    processed_images = []  # List to store processed image filenames

    if request.method == 'POST':
        if 'files' not in request.files:
            error_message = 'No file part'
        else:
            files = request.files.getlist('files')
            filter_type = request.form.get('filter', None)
            for file in files:
                if file.filename == '':
                    continue  # Skip files without a filename
                file_path = os.path.join('uploads', file.filename)
                file.save(file_path)

                # Apply filter if selected
                if filter_type:
                    try:
                        filtered_image = apply_filter(file_path, filter_type)
                        filtered_image.save(file_path)
                    except Exception as e:
                        error_message = f'Error applying filter: {str(e)}'
                        filtered_image = Image.open(file_path)  # Revert to original image if error

                try:
                    image, landmarks = get_landmarks(file_path)
                    if not landmarks:
                        error_message = 'No faces detected in some images'
                    else:
                        image_with_landmarks = draw_landmarks(image, landmarks)
                        output_filename = f'{os.path.splitext(file.filename)[0]}_output.jpg'
                        output_path = os.path.join('static', output_filename)
                        cv2.imwrite(output_path, image_with_landmarks)
                        processed_images.append(output_filename)
                except FileNotFoundError as e:
                    error_message = str(e)

    return render_template('index.html', error_message=error_message, processed_images=processed_images)

if __name__ == "__main__":
    app.run(debug=True)
