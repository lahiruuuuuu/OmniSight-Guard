import cv2
from ultralytics import YOLO
import streamlit as st
import time
import pandas as pd
from database import db_session, SafetyViolation, engine

# Page Config 
st.set_page_config(page_title="OmniSight Guard", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stSidebar { background-color: #161b22; }
    </style>
    """, unsafe_allow_html=True)

st.title("OmniSight Guard | AI Auditor")

# Sidebar Controls and live data
st.sidebar.header("Auditor Session Info")
company_name = st.sidebar.text_input("Company Name", "SafeLogistics Corp")
driver_name = st.sidebar.text_input("Driver Name", "Enter Driver Name...")
cooldown_val = st.sidebar.slider("Log Cooldown (Seconds)", 1, 10, 5)

# Safety 
run_monitor = st.sidebar.checkbox("Run Safety Monitor", value=True)

st.sidebar.divider()
st.sidebar.subheader("Live Audit Logs")
sidebar_table = st.sidebar.empty() # Placeholder for live-updating table

#  State management for Cooldown
if 'last_log_time' not in st.session_state:
    st.session_state.last_log_time = 0

#  AI logic 
model = YOLO('yolov8n.pt') 
cap = cv2.VideoCapture(0)
frame_placeholder = st.empty()

st.info(f"System Live: Monitoring {driver_name} from {company_name}...")


# add run_monitor so unchecking it stops the camera
while cap.isOpened() and run_monitor:
    ret, frame = cap.read()
    if not ret: 
        break

    results = model(frame, verbose=False)
    current_time = time.time()
    
    # Define red zone 
    rz_x1, rz_y1, rz_x2, rz_y2 = 100, 300, 500, 450
    cv2.rectangle(frame, (rz_x1, rz_y1), (rz_x2, rz_y2), (0, 0, 255), 3)
    cv2.putText(frame, "DANGER ZONE", (rz_x1, rz_y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) == 0: # Person detected
                x1, y1, x2, y2 = box.xyxy[0]
                cx, cy = int((x1 + x2) / 2), int(y2) # Feet position
                
                # Check if Feet are in the Red Zone
                if rz_x1 < cx < rz_x2 and rz_y1 < cy < rz_y2:
                    
                    if current_time - st.session_state.last_log_time > cooldown_val:
                        
                        violation = SafetyViolation(
                            violation_type=f"Red Zone Breach: {driver_name} ({company_name})", 
                            confidence=float(box.conf[0])
                        )
                        db_session.add(violation)
                        db_session.commit()
                        
                        st.session_state.last_log_time = current_time
                        st.sidebar.warning(f"LOGGED: {driver_name}")

    # Display the video
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame_rgb, channels="RGB")

    # Update sidebar table
    try:
        df_side = pd.read_sql("SELECT timestamp, violation_type FROM violations ORDER BY id DESC LIMIT 5", engine)
        sidebar_table.dataframe(df_side)
    except:
        sidebar_table.write("No logs yet.")

#  Final results
cap.release()
st.divider()
st.header("Final Audit Report & Data Export")

try:
    df_final = pd.read_sql("SELECT * FROM violations", engine)
    if not df_final.empty:
        st.write("Complete history of safety breaches:")
        st.dataframe(df_final.sort_values(by="timestamp", ascending=False), use_container_width=True)
        
        # Download Feature
        csv = df_final.to_csv(index=False).encode('utf-8')
        st.download_button("Download Full Report (CSV)", csv, "safety_audit.csv", "text/csv")
    else:
        st.info("No data recorded during this session.")
except Exception as e:
    st.error(f"Error loading database: {e}")