from flask import Flask, render_template, request, redirect, url_for
import face_recognition
import cv2
import numpy as np
import pandas as pd
import os
from datetime import datetime

app = Flask(__name__)

# Global variables to store known faces and names
known_face_encodings = []
known_face_names = []

def mark_attendance(name):
    # Ensure the 'attendance.csv' file exists
    if not os.path.isfile('attendance.csv'):
        df = pd.DataFrame(columns=['name', 'date', 'time'])
    else:
        df = pd.read_csv('attendance.csv')
    
    now = datetime.now()
    date = now.strftime('%Y-%m-%d')
    time = now.strftime('%H:%M:%S')
    df = df.append({'name': name, 'date': date, 'time': time}, ignore_index=True)
    df.to_csv('attendance.csv', index=False)

def load_known_faces():
    global known_face_encodings, known_face_names
    known_face_encodings = []
    known_face_names = []
    
    if not os.path.isfile('registered_users.csv'):
        print("No registered users file found.")
        return
    
    df = pd.read_csv('registered_users.csv')
    for _, row in df.iterrows():
        image_path = row['image_path']
        if os.path.exists(image_path) and image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = face_recognition.load_image_file(image_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Ensure image is in RGB format
            encoding = face_recognition.face_encodings(img_rgb)
            if encoding:
                known_face_encodings.append(encoding[0])
                known_face_names.append(f"{row['name']} ({row['roll_number']})")
        else:
            print(f"Skipping {image_path} (Not a valid image file or path)")

load_known_faces()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return "No image file part", 400
        
        file = request.files['image']
        if file.filename == '':
            return "No selected file", 400
        
        file_path = os.path.join("static", file.filename)
        file.save(file_path)

        # Open the saved file for processing
        img = face_recognition.load_image_file(file_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Ensure image is in RGB format

        face_locations = face_recognition.face_locations(img_rgb)
        face_encodings = face_recognition.face_encodings(img_rgb, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                mark_attendance(name)

            face_names.append(name)

        return render_template('index.html', uploaded_image=file.filename, face_names=face_names)
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        if 'image' not in request.files:
            return "No image file part", 400
        
        name = request.form['name']
        roll_number = request.form['roll_number']
        file = request.files['image']
        if file.filename == '':
            return "No selected file", 400
        
        file_path = os.path.join("known_faces", f"{name}_{roll_number}.jpg")
        file.save(file_path)

        if not os.path.isfile('registered_users.csv'):
            df = pd.DataFrame(columns=['name', 'roll_number', 'image_path'])
        else:
            df = pd.read_csv('registered_users.csv')
        
        df = df.append({'name': name, 'roll_number': roll_number, 'image_path': file_path}, ignore_index=True)
        df.to_csv('registered_users.csv', index=False)

        load_known_faces()

        return redirect(url_for('index'))
    return render_template('register.html')

@app.route('/test')
def test():
    return "Test page"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=4500)
