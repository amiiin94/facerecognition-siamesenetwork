import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from mtcnn import MTCNN
import pandas as pd
from datetime import datetime
import streamlit as st
from typing import Optional
from pathlib import Path
import random
import shutil
import time

# Initialize globals
@st.cache_resource
def load_detector():
    return MTCNN()

@st.cache_resource
def load_network():
    try:
        model = load_model('base_network.h5', compile=False)
        return model
    except FileNotFoundError:
        st.error("Error: base_network.h5 not found. Please ensure it is in the project root.")
        return None

# Set up paths and globals
face_database_path = "face_database"
embedding_database_file = "embedding_database.npy"
os.makedirs(face_database_path, exist_ok=True)

# Load the models
base_network = load_network()
detector = load_detector()

# Load or initialize embedding database
def load_embedding_database():
    embedding_database = {}
    if os.path.exists(embedding_database_file):
        try:
            embedding_database = np.load(embedding_database_file, allow_pickle=True).item()
            st.sidebar.success(f"Loaded embedding database with {len(embedding_database)} individuals.")
        except (EOFError, ValueError):
            st.sidebar.warning("Warning: embedding_database.npy is empty or corrupted. Initializing empty database.")
            embedding_database = {}
    else:
        st.sidebar.info("No embedding database found. Initializing empty database.")
    return embedding_database

# Helper functions
def load_and_preprocess_image(img_path: str) -> np.ndarray:
    img = cv2.imread(img_path)
    if img is None:
        return np.zeros((224, 224, 3), dtype=np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    # Use ResNet50 preprocessing instead of simple normalization
    img = preprocess_input(img)
    return img

def detect_and_crop_face(image: np.ndarray, size: tuple = (224, 224)) -> Optional[np.ndarray]:
    # Ensure image is uint8 for cvtColor
    if image.dtype != np.uint8:
        if image.max() <= 1.0:  # Image is normalized [0,1]
            image = (image * 255.0).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(img_rgb)
    if len(faces) == 0:
        return None
    face = faces[0]
    x, y, w, h = face['box']
    x, y = max(0, x), max(0, y)
    face_img = img_rgb[y:y+h, x:x+w]
    if face_img.size == 0:  # Check for empty crop
        return None
    face_img = cv2.resize(face_img, size)
    # Use ResNet50 preprocessing
    face_img = preprocess_input(face_img)
    return face_img

def compute_cosine_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
    dot_product = np.sum(emb1 * emb2, axis=-1)
    norm1 = np.sqrt(np.sum(emb1 * emb1, axis=-1))
    norm2 = np.sqrt(np.sum(emb2 * emb2, axis=-1))
    cosine_similarity = dot_product / (norm1 * norm2 + 1e-10)
    return (1 - cosine_similarity).item()

def log_attendance(person: str, csv_path: str = "attendance.csv") -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = {"Timestamp": timestamp, "Person": person}
    df = pd.DataFrame([entry])
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, mode='w', header=True, index=False)
    return timestamp

def identify_person(image: np.ndarray, threshold: float = 0.3) -> tuple[Optional[str], Optional[float], str]:
    face_img = detect_and_crop_face(image)
    if face_img is None:
        return None, None, "no_face"
    emb = base_network.predict(np.expand_dims(face_img, axis=0), verbose=0)[0]
    
    embedding_database = load_embedding_database()
    min_distance = float('inf')
    identified_person = None
    for person, embeddings in embedding_database.items():
        for db_emb in embeddings:
            distance = compute_cosine_distance(emb, db_emb)
            if distance < min_distance:
                min_distance = distance
                identified_person = person
    if min_distance > threshold:
        return None, min_distance, "no_match"
    return identified_person, min_distance, "match"

def crop_and_save_faces(images, person_name: str, output_dir: str = face_database_path) -> bool:
    person_output_path = os.path.join(output_dir, person_name)
    os.makedirs(person_output_path, exist_ok=True)
    
    st.text(f"Processing images for {person_name}...")
    valid_images = False
    
    for i, image_file in enumerate(images):
        img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), 1)
        if img is None:
            continue
        face_img = detect_and_crop_face(img)
        if face_img is None:
            st.warning(f"No face detected in image {i+1}")
            continue
        
        # Convert back from preprocessed to RGB for saving
        # Since preprocess_input transforms the image, we need to save a viewable version
        # We'll convert to uint8 for saving
        face_img_display = (face_img + 127.5).astype(np.uint8)  # Approximate reverse of preprocessing
        output_img_path = os.path.join(person_output_path, f"{person_name}_{i+1}.jpg")
        cv2.imwrite(output_img_path, cv2.cvtColor(face_img_display, cv2.COLOR_RGB2BGR))
        st.success(f"Saved cropped image: {output_img_path}")
        valid_images = True
    return valid_images

def add_person_to_database(person_name: str, max_images: int = 5) -> bool:
    person_path = os.path.join(face_database_path, person_name)
    if not os.path.exists(person_path):
        st.error(f"Error: No images found for {person_name} in {face_database_path}")
        return False
    
    images = [os.path.join(person_path, img) for img in os.listdir(person_path)]
    if not images:
        st.error(f"No valid images for {person_name}")
        return False
    
    images = random.sample(images, min(len(images), max_images))
    embeddings = []
    
    progress_bar = st.progress(0)
    for i, img_path in enumerate(images):
        img = load_and_preprocess_image(img_path)
        emb = base_network.predict(np.expand_dims(img, axis=0), verbose=0)[0]
        embeddings.append(emb)
        progress_bar.progress((i + 1) / len(images))
    
    embedding_database = load_embedding_database()
    embedding_database[person_name] = np.array(embeddings)
    np.save(embedding_database_file, embedding_database)
    st.success(f"Added {person_name} with {len(embeddings)} embeddings")
    return True

def view_attendance():
    st.subheader("Attendance Records")
    
    if os.path.exists("attendance.csv"):
        try:
            # Load the CSV file and display its structure first
            df = pd.read_csv("attendance.csv")
            
            # Show column names to help diagnose issues
            st.text("CSV Columns found: " + ", ".join(df.columns.tolist()))
            
            # Find the timestamp column (it might be named differently or have different capitalization)
            timestamp_col = None
            for col in df.columns:
                if col.lower() == "timestamp" or "time" in col.lower() or "date" in col.lower():
                    timestamp_col = col
                    break
            
            if timestamp_col is None:
                st.error("Could not find a timestamp column in the attendance CSV file.")
                st.dataframe(df.head())  # Show the first few rows to help diagnose
                return
                
            # Continue with the timestamp column we found
            unique_dates = pd.to_datetime(df[timestamp_col]).dt.date.unique()
            
            if unique_dates.size > 0:
                selected_date = st.selectbox(
                    "Select Date", 
                    options=sorted(unique_dates, reverse=True),
                    format_func=lambda x: x.strftime('%Y-%m-%d')
                )
                
                filtered_df = df[pd.to_datetime(df[timestamp_col]).dt.date == selected_date]
                
                if not filtered_df.empty:
                    st.write(f"### Attendance for {selected_date}")
                    
                    # Find the person column (similar to timestamp column)
                    person_col = None
                    for col in df.columns:
                        if col.lower() == "person" or "name" in col.lower() or "user" in col.lower():
                            person_col = col
                            break
                    
                    if person_col is None:
                        st.error("Could not find a person/name column in the attendance CSV file.")
                        st.dataframe(filtered_df)  # Show the filtered data as is
                        return
                    
                    # Group by person and show first and last entry
                    person_groups = filtered_df.groupby(person_col)
                    summary = []
                    
                    for person, group in person_groups:
                        times = pd.to_datetime(group[timestamp_col])
                        summary.append({
                            'Person': person,
                            'First Entry': times.min().strftime('%H:%M:%S'),
                            'Last Entry': times.max().strftime('%H:%M:%S'),
                            'Total Entries': len(group)
                        })
                    
                    summary_df = pd.DataFrame(summary)
                    st.dataframe(summary_df)
                    
                    # Show raw entries if requested
                    if st.checkbox("Show all entries"):
                        st.dataframe(filtered_df)
                    
                    # Download option
                    csv = filtered_df.to_csv(index=False)
                    st.download_button(
                        label="Download as CSV",
                        data=csv,
                        file_name=f"attendance_{selected_date}.csv",
                        mime="text/csv",
                    )
                else:
                    st.info(f"No attendance records for {selected_date}")
            else:
                st.info("No attendance records available")
        except Exception as e:
            st.error(f"Error processing attendance file: {str(e)}")
            st.info("Showing raw CSV data instead:")
            
            try:
                # Fallback to just displaying the raw CSV
                df = pd.read_csv("attendance.csv")
                st.dataframe(df)
            except Exception as inner_e:
                st.error(f"Could not read the CSV file: {str(inner_e)}")
    else:
        st.info("No attendance records available. Start taking attendance to create records.")

def capture_from_webcam(person_name: str, num_images: int = 5):
    st.subheader("Webcam Capture")
    
    # Create a directory to store temporary captured images
    temp_dir = "temp_webcam_images"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Clear the directory
    for file in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, file))
    
    captured_images = []
    
    # Video capture component
    picture = st.camera_input("Take a picture")
    
    if picture is not None:
        img_path = os.path.join(temp_dir, f"{person_name}_{len(os.listdir(temp_dir))+1}.jpg")
        with open(img_path, "wb") as f:
            f.write(picture.getbuffer())
        
        captured_images.append(picture)
        st.success(f"Captured image {len(captured_images)}/{num_images}")
        
        if len(captured_images) >= num_images:
            st.success(f"Captured all {num_images} images!")
            return captured_images
    
    return captured_images

def view_registered_people():
    st.subheader("Registered People")
    
    embedding_database = load_embedding_database()
    
    if not embedding_database:
        st.info("No people registered in the database")
        return
    
    people = list(embedding_database.keys())
    
    for person in people:
        col1, col2 = st.columns([3, 1])
        
        # Display person's name
        col1.write(person)
        
        # Add a delete button with confirmation
        if col2.button(f"Delete", key=f"delete_{person}"):
            # Store deletion confirmation in session state
            st.session_state[f"confirm_delete_{person}"] = True
        
        # Show confirmation dialog
        if st.session_state.get(f"confirm_delete_{person}", False):
            st.warning(f"Are you sure you want to delete {person}?")
            confirm_col1, confirm_col2 = st.columns(2)
            if confirm_col1.button("Yes", key=f"confirm_yes_{person}"):
                try:
                    # Remove from embedding database
                    del embedding_database[person]
                    np.save(embedding_database_file, embedding_database)
                    
                    # Remove image directory
                    person_path = os.path.join(face_database_path, person)
                    if os.path.exists(person_path):
                        shutil.rmtree(person_path)
                    
                    st.success(f"Deleted {person} from the database")
                    # Clear confirmation state
                    st.session_state[f"confirm_delete_{person}"] = False
                    st.rerun()
                except Exception as e:
                    st.error(f"Error deleting {person}: {e}")
            if confirm_col2.button("No", key=f"confirm_no_{person}"):
                st.session_state[f"confirm_delete_{person}"] = False
                
# Main app function
def main():
    # Check if the base network is loaded
    if base_network is None:
        st.error("Failed to load the base network model. Please ensure base_network.h5 is available.")
        return
    
    # App title and sidebar
    st.title("Face Recognition Attendance System")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose mode", 
        ["Home", "Register New Person", "Take Attendance", "View Attendance", "View Registered People"]
    )
    
    # Sidebar information
    st.sidebar.info(f"Database contains {len(load_embedding_database())} registered people")
    
    # Application modes
    if app_mode == "Home":
        st.header("Face Recognition Attendance System")
        st.write("""
        Welcome to the Face Recognition Attendance System! This application allows you to:
        
        1. Register new people for face recognition
        2. Take attendance using your webcam
        3. View attendance records
        4. Manage registered people
        
        Select a mode from the sidebar to get started.
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            st.info("💡 **Register New Person**: Add people to the face recognition database")
        with col2:
            st.info("📸 **Take Attendance**: Use your webcam to mark attendance")
        
        col3, col4 = st.columns(2)
        with col3:
            st.info("📊 **View Attendance**: Check attendance records and reports")
        with col4:
            st.info("👥 **View Registered People**: Manage people in the database")
    
    elif app_mode == "Register New Person":
        st.header("Register New Person")
        
        # Input for name
        person_name = st.text_input("Enter person's name")
        
        # Input method selection
        input_method = st.radio("Choose input method:", ["Upload Images", "Capture from Webcam"])
        
        if person_name:
            if input_method == "Upload Images":
                st.write("Upload 3-5 clear face images:")
                uploaded_files = st.file_uploader("Choose images", accept_multiple_files=True, type=["jpg", "jpeg", "png"])
                
                if uploaded_files and st.button("Process Images"):
                    if len(uploaded_files) < 3:
                        st.warning("Please upload at least 3 images for better recognition")
                    
                    with st.spinner("Processing images..."):
                        if crop_and_save_faces(uploaded_files, person_name):
                            if add_person_to_database(person_name):
                                st.success(f"Successfully added {person_name} to the database.")
                            else:
                                st.error(f"Failed to add {person_name} to the database.")
                        else:
                            st.error(f"No valid faces detected for {person_name}. Registration failed.")
            
            elif input_method == "Capture from Webcam":
                st.write("Capture 5 images using your webcam:")
                st.info("Take multiple pictures from slightly different angles for better recognition")
                
                # Initialize if not exists
                if "captured_images" not in st.session_state:
                    st.session_state.captured_images = []
                
                if len(st.session_state.captured_images) < 5:
                    picture = st.camera_input("Take a picture")
                    
                    if picture is not None and st.button("Save Image"):
                        st.session_state.captured_images.append(picture)
                        st.success(f"Captured image {len(st.session_state.captured_images)}/5")
                        st.rerun()
                
                # Display captured images
                if st.session_state.captured_images:
                    st.write(f"Captured {len(st.session_state.captured_images)}/5 images")
                    cols = st.columns(min(5, len(st.session_state.captured_images)))
                    
                    for i, col in enumerate(cols):
                        if i < len(st.session_state.captured_images):
                            col.image(st.session_state.captured_images[i], width=224)
                
                # Process images when enough are captured
                if len(st.session_state.captured_images) >= 5 and st.button("Register Person"):
                    with st.spinner("Processing images..."):
                        temp_dir = "temp_webcam_images"
                        os.makedirs(temp_dir, exist_ok=True)
                        
                        # Save captured images temporarily
                        for i, img in enumerate(st.session_state.captured_images):
                            img_path = os.path.join(temp_dir, f"{person_name}_{i+1}.jpg")
                            with open(img_path, "wb") as f:
                                f.write(img.getbuffer())
                        
                        if crop_and_save_faces(st.session_state.captured_images, person_name):
                            if add_person_to_database(person_name):
                                st.success(f"Successfully added {person_name} to the database.")
                                # Clear session state
                                st.session_state.captured_images = []
                            else:
                                st.error(f"Failed to add {person_name} to the database.")
                        else:
                            st.error(f"No valid faces detected for {person_name}. Registration failed.")
        else:
            st.warning("Please enter a name to register")
    
    elif app_mode == "Take Attendance":
        st.header("Take Attendance")
        
        # Threshold slider
        threshold = st.slider("Recognition Threshold", 0.1, 0.99, 0.3, 0.05,
                             help="Lower values make recognition stricter (fewer false positives but might miss people)")
        
        # Initialize session state for attendance log
        if 'attendance_log' not in st.session_state:
            st.session_state.attendance_log = []
        
        if 'last_logged' not in st.session_state:
            st.session_state.last_logged = {}
        
        # Check if embedding database is empty
        if not load_embedding_database():
            st.error("No people registered in the database. Please register people first.")
        else:
            # Status placeholders
            status_placeholder = st.empty()
            
            # Create a video capture object
            camera_input = st.camera_input("Capture for Attendance")
            
            if camera_input:
                # Process the image
                image_bytes = camera_input.getvalue()
                image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
                
                # Identify person
                person, distance, status = identify_person(image, threshold)
                
                if status == "no_face":
                    status_placeholder.warning("No face detected. Please try again.")
                elif status == "no_match":
                    if distance:
                        status_placeholder.warning(f"No match found. Distance: {distance:.4f}")
                    else:
                        status_placeholder.warning("No match found.")
                elif person:
                    current_time = datetime.now()
                    # Check if this person was logged recently (within 60 seconds)
                    if person not in st.session_state.last_logged or (current_time - st.session_state.last_logged[person]).total_seconds() > 60:
                        timestamp = log_attendance(person)
                        st.session_state.last_logged[person] = current_time
                        
                        # Add to session log
                        log_entry = f"{timestamp} - {person} (Distance: {distance:.4f})"
                        st.session_state.attendance_log.append(log_entry)
                        
                        # Show success message
                        status_placeholder.success(f"Attendance logged for {person}!")
                    else:
                        # Already logged recently
                        status_placeholder.info(f"Already logged for {person} recently. Distance: {distance:.4f}")
            
            # Display attendance log
            if st.session_state.attendance_log:
                st.subheader("Session Attendance Log")
                for log in reversed(st.session_state.attendance_log[-10:]):
                    st.text(log)
    
    elif app_mode == "View Attendance":
        view_attendance()
    
    elif app_mode == "View Registered People":
        view_registered_people()

if __name__ == "__main__":
    main()