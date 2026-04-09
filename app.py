import numpy as np
import pandas as pd
import joblib
import streamlit as st
model=joblib.load("ai_model.pkl")
scaler_x=joblib.load("scaler_x.pkl")
scaler_y=joblib.load("scaler_y.pkl")
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,r2_score
import streamlit.components.v1 as components

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
        accel_x/9.8, accel_y/9.8, accel_z/9.8,
        gyro_x, gyro_y, gyro_z,
        mag_x, mag_y, mag_z
    ]

    sample_array = np.array(sample).reshape(1, -1)

    if np.all(sample_array == 0):
        roll, pitch, yaw = 0.0, 0.0, 0.0
    else:
      sample_array_scaled = scaler_x.transform(sample_array)
      y_pred_scaled = model.predict(sample_array_scaled)
      y_pred = scaler_y.inverse_transform(y_pred_scaled)
      roll = -y_pred[0][0]
      pitch = -y_pred[0][1]
      yaw_sin = y_pred[0][2]
      yaw_cos = y_pred[0][3]
      yaw = np.arctan2(yaw_sin, yaw_cos)
      yaw =-yaw

    quat = euler_to_quaternion(roll, pitch, yaw)
    dcm = euler_to_dcm(roll, pitch, yaw)
    with right_col:
            st.success("Prediction Completed Successfully")

            st.subheader("Results")

            st.markdown("### Euler Angles")
            st.write(f"**Roll:** {roll:.6f}")
            st.write(f"**Pitch:** {pitch:.6f}")
            st.write(f"**Yaw:** {yaw:.6f}")
            st.markdown("### 3D Aircraft Model")

            with open("low_poly_airplane.glb", "rb") as f:
                glb_data = f.read()

            components.html(f"""
            <html>
            <head>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/three@0.128/examples/js/loaders/GLTFLoader.js"></script>
            </head>
            <body style="margin:0; background-color:white;">
            <script>
                const scene = new THREE.Scene();
                scene.background = new THREE.Color(0xffffff);

                const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 1000);
                camera.position.set(0, 8, 50);
                camera.lookAt(0, 0, 0);

                const renderer = new THREE.WebGLRenderer({{antialias:true, alpha:true}});
                renderer.setSize(450, 450);
                document.body.appendChild(renderer.domElement);

                const ambientLight = new THREE.AmbientLight(0xffffff, 1.8);
                scene.add(ambientLight);

                const dirLight1 = new THREE.DirectionalLight(0xffffff, 1.2);
                dirLight1.position.set(5, 5, 5);
                scene.add(dirLight1);

                const dirLight2 = new THREE.DirectionalLight(0xffffff, 1.0);
                dirLight2.position.set(-5, 3, 5);
                scene.add(dirLight2);

                const loader = new THREE.GLTFLoader();

                const blob = new Blob([new Uint8Array({list(glb_data)})], {{type: 'model/gltf-binary'}});
                const url = URL.createObjectURL(blob);

                loader.load(url, function(gltf) {{
                    const model = gltf.scene;

                    model.scale.set(0.4, 0.4, 0.4);
                    model.position.set(0, 0, 0);

                    const box = new THREE.Box3().setFromObject(model);
                    const center = box.getCenter(new THREE.Vector3());
                    model.position.x -= center.x;
                    model.position.y -= center.y;
                    model.position.z -= center.z;

                    scene.add(model);

                    function animate() {{
                       requestAnimationFrame(animate);
                       model.rotation.y += 0.01;
                       renderer.render(scene, camera);
                       }}

                       animate();
                }});
            </script>
            </body>
            </html>
            """, height=450)
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