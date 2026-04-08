import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor


# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="AI Flight Orientation System",
    page_icon="🚀",
    layout="wide"
)


# =========================
# Load Data & Train Model
# =========================
@st.cache_resource
def load_and_train_model():
    df = pd.read_csv("imu_data.csv")
    df = df.head(20000)

    features = [
        "accel_x", "accel_y", "accel_z",
        "gyro_x", "gyro_y", "gyro_z",
        "mag_x", "mag_y", "mag_z"
    ]

    targets = ["roll", "pitch", "yaw"]

    X = df[features]
    y = df[targets]

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)

    return model


model = load_and_train_model()


# =========================
# Conversion Functions
# =========================
def euler_to_quaternion(roll, pitch, yaw):
    roll = np.radians(roll)
    pitch = np.radians(pitch)
    yaw = np.radians(yaw)

    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return (w, x, y, z)


def euler_to_dcm(roll, pitch, yaw):
    roll = np.radians(roll)
    pitch = np.radians(pitch)
    yaw = np.radians(yaw)

    cr = np.cos(roll)
    sr = np.sin(roll)

    cp = np.cos(pitch)
    sp = np.sin(pitch)

    cy = np.cos(yaw)
    sy = np.sin(yaw)

    dcm = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr]
    ])

    return dcm


# =========================
# Sidebar
# =========================
st.sidebar.title("About Project")
st.sidebar.info(
    "This system uses IMU sensor values to predict flight orientation "
    "(Roll, Pitch, Yaw) using AI, then converts the results into "
    "Quaternion and Direction Cosine Matrix (DCM)."
)

st.sidebar.markdown("### Model Input")
st.sidebar.write("9 IMU sensor values:")
st.sidebar.write("- Accelerometer (x, y, z)")
st.sidebar.write("- Gyroscope (x, y, z)")
st.sidebar.write("- Magnetometer (x, y, z)")

st.sidebar.markdown("### Output")
st.sidebar.write("- Euler Angles")
st.sidebar.write("- Quaternion")
st.sidebar.write("- DCM")


# =========================
# Main Title
# =========================
st.markdown("## 🚀 AI Flight Orientation System")
st.caption("Enter IMU sensor values to estimate orientation using AI.")


# =========================
# Layout
# =========================
left_col, right_col = st.columns(2)


# =========================
# Left Column - Inputs
# =========================
with left_col:
    st.subheader("Sensor Inputs")

    accel_x = st.number_input("accel_x", value=0.0, format="%.6f")
    accel_y = st.number_input("accel_y", value=0.0, format="%.6f")
    accel_z = st.number_input("accel_z", value=0.0, format="%.6f")

    gyro_x = st.number_input("gyro_x", value=0.0, format="%.6f")
    gyro_y = st.number_input("gyro_y", value=0.0, format="%.6f")
    gyro_z = st.number_input("gyro_z", value=0.0, format="%.6f")

    mag_x = st.number_input("mag_x", value=0.0, format="%.6f")
    mag_y = st.number_input("mag_y", value=0.0, format="%.6f")
    mag_z = st.number_input("mag_z", value=0.0, format="%.6f")

    st.markdown("### Current Input Values")
    st.write([
        accel_x, accel_y, accel_z,
        gyro_x, gyro_y, gyro_z,
        mag_x, mag_y, mag_z
    ])

    predict = st.button("Predict Orientation", use_container_width=True)


# =========================
# Right Column - Results
# =========================
if predict:
    sample = [
        accel_x, accel_y, accel_z,
        gyro_x, gyro_y, gyro_z,
        mag_x, mag_y, mag_z
    ]

    sample_array = np.array(sample).reshape(1, -1)

    pred = model.predict(sample_array)[0]
    roll, pitch, yaw = pred

    quat = euler_to_quaternion(roll, pitch, yaw)
    dcm = euler_to_dcm(roll, pitch, yaw)

    with right_col:
        st.success("Prediction Completed Successfully")

        st.subheader("Results")

        st.markdown("### Euler Angles")
        st.write(f"**Roll:** {roll:.6f}")
        st.write(f"**Pitch:** {pitch:.6f}")
        st.write(f"**Yaw:** {yaw:.6f}")

        st.markdown("### Quaternion")
        quat_df = pd.DataFrame({
            "Component": ["w", "x", "y", "z"],
            "Value": [f"{q:.10f}" for q in quat]
        })
        st.dataframe(quat_df, use_container_width=True, hide_index=True)

        st.markdown("### DCM")
        dcm_df = pd.DataFrame(dcm, columns=["C1", "C2", "C3"], index=["R1", "R2", "R3"])
        st.dataframe(dcm_df, use_container_width=True)

else:
    with right_col:
        st.info("Results will appear here after clicking 'Predict Orientation'.")