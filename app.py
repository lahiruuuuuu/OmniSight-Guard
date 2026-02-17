import cv2
from ultralytics import YOLO
import streamlit as st
from database import db_session, SafetyViolation 

#  Page Config 
st.set_page_config(page_title="OmniSight Guard", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ OmniSight Guard | Level 6 AI Auditor")

#  Logic to log to SQL
def log_to_db(v_type, conf):
    violation = SafetyViolation(violation_type=v_type, confidence=float(conf))
    db_session.add(violation)
    db_session.commit()

#  AI inference loop
model = YOLO('yolov8n.pt') 
cap = cv2.VideoCapture(0)
frame_placeholder = st.empty()

st.info("System Live: Monitoring for Red Zone Intrusion...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    results = model(frame, verbose=False)
    
    for r in results:
        # Drawing a simulated red zone box
        cv2.rectangle(frame, (100, 300), (500, 450), (0, 0, 255), 2)
        
        # Check detections
        for box in r.boxes:
            if int(box.cls[0]) == 0: # If it's a person
                log_to_db("Personnel in Restricted Area", box.conf[0])

    # Display in Streamlit
    frame_rgb = cv2.cvtColor(r.plot(), cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame_rgb, channels="RGB")