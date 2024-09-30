import cv2
import dlib
import subprocess
import os
from pathlib import Path

def process_single_image(input_file_path, output_file_path, face_crop_percent, output_size):
    img = cv2.imread(str(input_file_path))
    
    face_detector = dlib.get_frontal_face_detector()
    faces = face_detector(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    if len(faces) > 0:
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        shape = predictor(img, faces[0])
        
        nose_tip = (shape.part(33).x, shape.part(33).y)
        
        crop_size = int(img.shape[1] * (face_crop_percent / 100))
        left = max(0, nose_tip[0] - crop_size // 2)
        top = max(0, nose_tip[1] - crop_size // 2)
        right = min(img.shape[1], left + crop_size)
        bottom = min(img.shape[0], top + crop_size)
        
        cropped = img[top:bottom, left:right]
        
        temp_file = "temp_cropped.jpg"
        cv2.imwrite(temp_file, cropped, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        ffmpeg_command = [
            "ffmpeg",
            "-y",
            "-i", temp_file,
            "-vf", f"scale={output_size}:{output_size}:force_original_aspect_ratio=decrease,pad={output_size}:{output_size}:(ow-iw)/2:(oh-ih)/2",
            "-q:v", "2",
            str(output_file_path)
        ]
        
        subprocess.run(ffmpeg_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        os.remove(temp_file)
        
        return True
    
    return False

def process_all_images(input_folder, output_folder, face_crop_percent, output_size):
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    for img_file in input_path.glob("*"):
        if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            output_file = output_path / f"out_{img_file.name}"
            success = process_single_image(img_file, output_file, face_crop_percent, output_size)
            if success:
                print(f"処理成功: {img_file.name}")
            else:
                print(f"顔検出失敗: {img_file.name}")

def main():
    input_folder = "input"
    output_folder = "output"
    face_crop_percent = 70
    output_size = 480
    
    process_all_images(input_folder, output_folder, face_crop_percent, output_size)

if __name__ == "__main__":
    main()