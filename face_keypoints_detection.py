import cv2
import dlib
import os

# Load pre-trained face detector and facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def get_landmarks(image_path):
    # Check if the image file exists
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file '{image_path}' not found or unable to open.")
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image '{image_path}'.")
    
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

def main(image_path):
    image, landmarks = get_landmarks(image_path)
    image_with_landmarks = draw_landmarks(image, landmarks)
    cv2.imshow('Facial Landmarks', image_with_landmarks)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Update path to your image file if needed
    image_path = 'test_image.jpg'  # Ensure this matches the renamed image file
    main(image_path)
