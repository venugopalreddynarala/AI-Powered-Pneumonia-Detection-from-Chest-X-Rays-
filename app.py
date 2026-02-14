"""
Streamlit web application for AI-powered pneumonia detection.
Provides interactive interface for X-ray upload, prediction, and visualization.
"""

import streamlit as st
import streamlit.components.v1 as components
import torch
import numpy as np
import cv2
from PIL import Image
import pandas as pd
import os
import base64
import json
from string import Template
from pathlib import Path

from utils.data_prep import preprocess_single_image
from utils.gradcam import (
    generate_gradcam_visualization, 
)
from utils.recommendations import (
    SeverityClassifier, 
    generate_recommendations,
    get_severity_color,
    generate_patient_report
)
from utils.metrics import compute_classification_metrics, plot_confusion_matrix, plot_roc_curve
from train import build_model


# Page configuration
st.set_page_config(
    page_title="AI Pneumonia Detection",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        padding: 1rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .normal-box {
        background-color: #E8F5E9;
        border-left: 5px solid #4CAF50;
    }
    .pneumonia-box {
        background-color: #FFEBEE;
        border-left: 5px solid #F44336;
    }
    .metric-card {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(model_path: str = 'models/model_weights.pth'):
    """Load trained model (cached for performance)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not os.path.exists(model_path):
        st.error(f"Model weights not found at {model_path}")
        st.info("Please train the model first using: `python train.py`")
        return None, device
    
    model = build_model(num_classes=2, pretrained=False)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, device


def predict_image(model, image_path, device):
    """
    Run prediction on uploaded image.
    
    Returns:
        Dictionary with prediction results
    """
    # Preprocess image
    input_tensor = preprocess_single_image(image_path).to(device)
    
    # Get prediction
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        confidence = probs[0].cpu().numpy()
        prediction = output.argmax(dim=1).item()
    
    return {
        'prediction': prediction,
        'confidence': confidence,
        'pneumonia_prob': confidence[1],
        'normal_prob': confidence[0]
    }


def display_prediction_results(pred_results):
    """Display prediction results in a formatted box."""
    prediction = pred_results['prediction']
    confidence = pred_results['pneumonia_prob'] if prediction == 1 else pred_results['normal_prob']
    
    if prediction == 1:
        st.markdown(f"""
        <div class="prediction-box pneumonia-box">
            <h2 style="color: #D32F2F;">⚠️ Pneumonia Detected</h2>
            <h3>Confidence: {confidence*100:.2f}%</h3>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="prediction-box normal-box">
            <h2 style="color: #388E3C;">✅ Normal</h2>
            <h3>Confidence: {confidence*100:.2f}%</h3>
        </div>
        """, unsafe_allow_html=True)
    
    # Probability chart
    prob_df = pd.DataFrame({
        'Class': ['Normal', 'Pneumonia'],
        'Probability': [pred_results['normal_prob'], pred_results['pneumonia_prob']]
    })
    
    st.bar_chart(prob_df.set_index('Class'))


def render_lung_glb_viewer(glb_path: str, severity_intensity: float, height: int = 650):
    try:
        severity_intensity = float(severity_intensity)
    except Exception:
        severity_intensity = 0.0
    severity_intensity = max(0.0, min(1.0, severity_intensity))

    normalized_glb_path = Path(str(glb_path).lstrip("/\\"))

    glb_read_error = ''
    try:
        glb_bytes = normalized_glb_path.read_bytes()
        glb_data_uri = (
            'data:model/gltf-binary;base64,'
            + base64.b64encode(glb_bytes).decode('utf-8')
        )
    except Exception as e:
        glb_data_uri = ''
        glb_read_error = str(e)

    html_template = Template("""<!doctype html>
<html lang=\"en\">
  <head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <style>
      html, body { margin: 0; padding: 0; width: 100%; height: 100%; background: #0b0b10; overflow: hidden; }
      #root { width: 100%; height: 100%; }
      canvas { display: block; width: 100%; height: 100%; }
    </style>
  </head>
  <body>
    <div id=\"root\"></div>

    <script type=\"importmap\">
      {
        \"imports\": {
          \"three\": \"https://unpkg.com/three@0.160.0/build/three.module.js\",
          \"three/addons/\": \"https://unpkg.com/three@0.160.0/examples/jsm/\"
        }
      }
    </script>

    <script type=\"module\">
      import * as THREE from 'three';
      import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
      import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

      const glbUrl = $glb_url;
      const severityIntensity = $severity_intensity;
      const glbReadError = $glb_read_error;

      const root = document.getElementById('root');
      if (!glbUrl) {
        console.error('Failed to load GLB lung model: unable to read GLB bytes', glbReadError);
        root.innerHTML = '';
      } else {
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x0b0b10);

        const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 1000);

        const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
        renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
        renderer.setClearColor(0x0b0b10, 1);
        root.appendChild(renderer.domElement);

        const ambient = new THREE.AmbientLight(0xffffff, 0.65);
        scene.add(ambient);

        const dir = new THREE.DirectionalLight(0xffffff, 0.75);
        dir.position.set(2.0, 3.0, 4.0);
        scene.add(dir);

        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.06;
        controls.rotateSpeed = 0.6;
        controls.zoomSpeed = 0.8;
        controls.panSpeed = 0.6;

        let stop = false;

        function resize() {
          const w = root.clientWidth || 1;
          const h = root.clientHeight || 1;
          renderer.setSize(w, h, false);
          camera.aspect = w / h;
          camera.updateProjectionMatrix();
        }

        window.addEventListener('resize', resize);
        resize();

        const loader = new GLTFLoader();
        loader.load(
          glbUrl,
          (gltf) => {
            const model = gltf.scene;

            const overlayVertexShader = `
              varying vec3 vWorldPos;
              varying vec3 vNormalW;
              varying vec2 vUv;
              void main() {
                vec4 worldPos = modelMatrix * vec4(position, 1.0);
                vWorldPos = worldPos.xyz;
                vNormalW = normalize(mat3(modelMatrix) * normal);
                vUv = uv;
                gl_Position = projectionMatrix * viewMatrix * worldPos;
              }
            `;

            const overlayFragmentShader = `
              precision highp float;
              varying vec3 vWorldPos;
              varying vec3 vNormalW;
              varying vec2 vUv;
              uniform vec3 uBaseColor;
              uniform sampler2D uBaseMap;
              uniform float uHasMap;
              uniform float uSeverity;

              float hash31(vec3 p) {
                p = fract(p * 0.1031);
                p += dot(p, p.yzx + 33.33);
                return fract((p.x + p.y) * p.z);
              }

              float noise3(vec3 p) {
                vec3 i = floor(p);
                vec3 f = fract(p);
                f = f * f * (3.0 - 2.0 * f);

                float n000 = hash31(i + vec3(0.0, 0.0, 0.0));
                float n100 = hash31(i + vec3(1.0, 0.0, 0.0));
                float n010 = hash31(i + vec3(0.0, 1.0, 0.0));
                float n110 = hash31(i + vec3(1.0, 1.0, 0.0));
                float n001 = hash31(i + vec3(0.0, 0.0, 1.0));
                float n101 = hash31(i + vec3(1.0, 0.0, 1.0));
                float n011 = hash31(i + vec3(0.0, 1.0, 1.0));
                float n111 = hash31(i + vec3(1.0, 1.0, 1.0));

                float nx00 = mix(n000, n100, f.x);
                float nx10 = mix(n010, n110, f.x);
                float nx01 = mix(n001, n101, f.x);
                float nx11 = mix(n011, n111, f.x);

                float nxy0 = mix(nx00, nx10, f.y);
                float nxy1 = mix(nx01, nx11, f.y);

                return mix(nxy0, nxy1, f.z);
              }

              float fbm(vec3 p) {
                float v = 0.0;
                float a = 0.5;
                for (int i = 0; i < 5; i++) {
                  v += a * noise3(p);
                  p *= 2.02;
                  a *= 0.5;
                }
                return v;
              }

              vec3 severityColor(float s) {
                if (s < 0.4) return vec3(1.0, 0.8353, 0.3098);
                if (s < 0.7) return vec3(1.0, 0.5961, 0.0);
                return vec3(0.8980, 0.2235, 0.2078);
              }

              void main() {
                float s = clamp((uSeverity - 0.2) / 0.8, 0.0, 1.0);
                vec3 baseCol = uBaseColor;
                if (uHasMap > 0.5) {
                  baseCol *= texture2D(uBaseMap, vUv).rgb;
                }
                if (s <= 0.0) {
                  vec3 lightDir0 = normalize(vec3(0.4, 0.8, 0.6));
                  float ndl0 = clamp(dot(normalize(vNormalW), lightDir0), 0.0, 1.0);
                  float shade0 = 0.55 + 0.45 * ndl0;
                  gl_FragColor = vec4(baseCol * shade0, 1.0);
                  return;
                }

                vec3 p = vWorldPos * 0.75;
                float n = fbm(p);

                float coverage = mix(0.86, 0.42, s);
                float softness = mix(0.06, 0.22, s);
                float m = smoothstep(coverage, coverage + softness, n);
                m = pow(m, mix(2.4, 1.1, s));
                m *= s;

                vec3 infection = severityColor(uSeverity);
                vec3 mixed = mix(baseCol, infection, m);

                vec3 lightDir = normalize(vec3(0.4, 0.8, 0.6));
                float ndl = clamp(dot(normalize(vNormalW), lightDir), 0.0, 1.0);
                float shade = 0.45 + 0.55 * ndl;

                gl_FragColor = vec4(mixed * shade, 1.0);
              }
            `;

            model.traverse((obj) => {
              if (obj && obj.isMesh) {
                const origMat = obj.material;
                const base = (origMat && origMat.color)
                  ? origMat.color.clone()
                  : new THREE.Color(0.82, 0.82, 0.82);
                const baseMap = (origMat && origMat.map) ? origMat.map : null;

                const side = origMat && typeof origMat.side === 'number'
                  ? origMat.side
                  : THREE.FrontSide;

                obj.material = new THREE.ShaderMaterial({
                  uniforms: {
                    uBaseColor: { value: base },
                    uBaseMap: { value: baseMap },
                    uHasMap: { value: baseMap ? 1.0 : 0.0 },
                    uSeverity: { value: severityIntensity }
                  },
                  vertexShader: overlayVertexShader,
                  fragmentShader: overlayFragmentShader,
                  transparent: false,
                  depthWrite: true,
                  depthTest: true,
                  side
                });
              }
            });

            scene.add(model);

            const box = new THREE.Box3().setFromObject(model);
            const size = new THREE.Vector3();
            box.getSize(size);
            const center = new THREE.Vector3();
            box.getCenter(center);

            const maxDim = Math.max(size.x, size.y, size.z) || 1;
            camera.position.set(center.x, center.y, center.z + maxDim * 2.2);
            camera.near = Math.max(0.01, maxDim / 1000);
            camera.far = maxDim * 50;
            camera.updateProjectionMatrix();

            controls.target.copy(center);
            controls.update();
          },
          undefined,
          (err) => {
            console.error('Failed to load GLB lung model:', err);
            stop = true;
            window.removeEventListener('resize', resize);
            root.innerHTML = '';
          }
        );

        function animate() {
          if (stop) return;
          controls.update();
          renderer.render(scene, camera);
          requestAnimationFrame(animate);
        }
        animate();
      }
    </script>
  </body>
</html>""")

    html = html_template.substitute(
        glb_url=json.dumps(glb_data_uri if glb_data_uri else None),
        severity_intensity=json.dumps(severity_intensity),
        glb_read_error=json.dumps(glb_read_error)
    )

    components.html(html, height=height, scrolling=False)


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<p class="main-header">🫁 AI-Powered Pneumonia Detection System</p>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("📋 Navigation")
    app_mode = st.sidebar.selectbox(
        "Select Mode",
        ["Prediction", "Model Evaluation", "About"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **How to use:**
    1. Upload a chest X-ray image
    2. View prediction results
    3. Analyze Grad-CAM visualization
    4. Explore 3D lung mapping
    5. Read medical recommendations
    """)
    
    # Load model
    model, device = load_model()
    
    if model is None:
        st.stop()
    
    # PREDICTION MODE
    if app_mode == "Prediction":
        st.header("🔍 X-Ray Analysis")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload Chest X-Ray Image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a chest X-ray image for pneumonia detection"
        )
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Display original image
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📷 Original X-Ray")
                image = Image.open(temp_path)
                st.image(image,  use_container_width=True)
            
            # Run prediction
            with st.spinner("🔬 Analyzing X-ray..."):
                pred_results = predict_image(model, temp_path, device)
            
            with col2:
                st.subheader("🎯 Prediction Results")
                display_prediction_results(pred_results)
            
            st.markdown("---")
            
            # Grad-CAM Visualization
            st.header("🔥 Grad-CAM Heatmap Analysis")
            
            with st.spinner("Generating Grad-CAM visualization..."):
                try:
                    # Load and preprocess image
                    original_img = cv2.imread(temp_path)
                    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                    original_img = cv2.resize(original_img, (224, 224))
                    
                    input_tensor = preprocess_single_image(temp_path).to(device)
                    
                    # Generate Grad-CAM
                    heatmap, overlaid, cam_intensity = generate_gradcam_visualization(
                        model, input_tensor, original_img
                    )
                    
                    # Display Grad-CAM
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("🔥 Heatmap Overlay")
                        st.image(overlaid,  use_container_width=True)
                        st.caption("Red regions indicate areas of high model attention")
                    
                    with col2:
                        st.subheader("📊 Intensity Analysis")
                        
                        # Create simple heatmap for visualization
                        heatmap_normalized = cv2.resize(
                            (cam_intensity * np.ones((224, 224)) * 255).astype(np.uint8),
                            (224, 224)
                        )
                        st.image(heatmap_normalized,  use_container_width=True, 
                                caption="Activation intensity map")
                    
                    # Severity Classification
                    st.markdown("---")
                    st.header("📈 Severity Assessment")
                    
                    severity_classifier = SeverityClassifier()
                    affected_area = severity_classifier.get_affected_area_percentage(
                        cam_intensity * np.ones((224, 224))
                    )
                    
                    severity = severity_classifier.classify(
                        pred_results['pneumonia_prob'],
                        affected_area
                    )
                    
                    severity_color = get_severity_color(severity)
                    
                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>Severity Level</h4>
                            <h2 style="color: {severity_color};">{severity}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>Confidence</h4>
                            <h2>{pred_results['pneumonia_prob']*100:.1f}%</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>Affected Area</h4>
                            <h2>{affected_area:.1f}%</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # 3D Visualization
                    st.markdown("---")
                    st.header("🫁 3D Lung Visualization")
                    
                    with st.spinner("Loading anatomical lung model..."):
                        severity_intensity = float(pred_results['pneumonia_prob']) * (float(affected_area) / 100.0)
                        render_lung_glb_viewer("/public/models/lung_carcinoma.glb", severity_intensity)
                    
                    # Medical Recommendations
                    st.markdown("---")
                    st.header("💊 Medical Recommendations")
                    
                    recommendations = generate_recommendations(
                        severity,
                        pred_results['pneumonia_prob'],
                        affected_area
                    )
                    
                    # Urgency Alert
                    if recommendations['urgency_level'] in ['High', 'Critical']:
                        st.error(f"⚠️ {recommendations['urgency_level']} Priority - {recommendations['follow_up']}")
                    elif recommendations['urgency_level'] == 'Moderate':
                        st.warning(f"⚠️ {recommendations['follow_up']}")
                    else:
                        st.info(f"ℹ️ {recommendations['follow_up']}")
                    
                    # Recommendations list
                    st.subheader("📋 Action Items:")
                    for i, rec in enumerate(recommendations['recommendations'], 1):
                        st.write(f"{i}. {rec}")
                    
                    # Warning box
                    if recommendations['warning']:
                        st.warning(f"⚠️ **Warning:** {recommendations['warning']}")
                    
                    # Disclaimer
                    st.info(f"ℹ️ **Disclaimer:** {recommendations['disclaimer']}")
                    
                    # Download Report
                    st.markdown("---")
                    report_text = generate_patient_report(
                        severity,
                        pred_results['pneumonia_prob'],
                        affected_area,
                        recommendations
                    )
                    
                    st.download_button(
                        label="📄 Download Full Report",
                        data=report_text,
                        file_name="pneumonia_analysis_report.txt",
                        mime="text/plain"
                    )
                    
                except Exception as e:
                    st.error(f"Error during visualization: {e}")
                    st.exception(e)
            
            # Cleanup
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    # EVALUATION MODE
    elif app_mode == "Model Evaluation":
        st.header("📊 Model Performance Evaluation")
        
        results_dir = "results"
        
        if not os.path.exists(results_dir):
            st.warning("⚠️ No evaluation results found. Please run evaluation first:")
            st.code("python evaluate.py", language="bash")
        else:
            # Load metrics
            metrics_path = os.path.join(results_dir, "metrics.csv")
            if os.path.exists(metrics_path):
                st.subheader("📈 Performance Metrics")
                
                metrics_df = pd.read_csv(metrics_path)
                
                # Display metrics in columns
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("Accuracy", f"{metrics_df['accuracy'].values[0]:.4f}")
                with col2:
                    st.metric("Precision", f"{metrics_df['precision'].values[0]:.4f}")
                with col3:
                    st.metric("Recall", f"{metrics_df['recall'].values[0]:.4f}")
                with col4:
                    st.metric("F1-Score", f"{metrics_df['f1_score'].values[0]:.4f}")
                with col5:
                    st.metric("AUC-ROC", f"{metrics_df['auc_roc'].values[0]:.4f}")
                
                st.markdown("---")
                
                # Display visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🔲 Confusion Matrix")
                    cm_path = os.path.join(results_dir, "confusion_matrix.png")
                    if os.path.exists(cm_path):
                        st.image(cm_path,  use_container_width=True)
                
                with col2:
                    st.subheader("📈 ROC Curve")
                    roc_path = os.path.join(results_dir, "roc_curve.png")
                    if os.path.exists(roc_path):
                        st.image(roc_path,  use_container_width=True)
                
                # Metrics bar chart
                st.subheader("📊 Metrics Overview")
                metrics_bar_path = os.path.join(results_dir, "metrics_bar.png")
                if os.path.exists(metrics_bar_path):
                    st.image(metrics_bar_path,  use_container_width=True)
                
                # Classification report
                st.subheader("📋 Detailed Classification Report")
                report_path = os.path.join(results_dir, "classification_report.txt")
                if os.path.exists(report_path):
                    with open(report_path, 'r') as f:
                        report_text = f.read()
                    st.text(report_text)
    
    # ABOUT MODE
    else:
        st.header("ℹ️ About This System")
        
        st.markdown("""
        ### AI-Powered Pneumonia Detection System
        
        This application uses deep learning to detect pneumonia from chest X-ray images.
        
        **Key Features:**
        - 🔬 **DenseNet121 Architecture**: State-of-the-art CNN pretrained on ImageNet
        - 🔥 **Grad-CAM Visualization**: Explainable AI showing decision-making regions
        - 🫁 **3D Lung Mapping**: Interactive 3D visualization of affected areas
        - 📊 **Comprehensive Metrics**: Accuracy, Precision, Recall, F1, AUC-ROC, BLEU
        - 💊 **Medical Recommendations**: Severity-based actionable advice
        
        **Technology Stack:**
        - Deep Learning: PyTorch, TorchVision
        - Visualization: Plotly, Matplotlib, Seaborn
        - Web Framework: Streamlit
        - Metrics: Scikit-learn, NLTK
        
        **Model Performance:**
        - Training Dataset: Kaggle Chest X-Ray Pneumonia Dataset
        - Classes: Normal vs Pneumonia (Binary Classification)
        - Expected Accuracy: >92%
        - Expected AUC-ROC: >0.95
        
        **Disclaimer:**
        This system is designed for educational and screening purposes only. 
        It should NOT be used as a substitute for professional medical diagnosis.
        Always consult qualified healthcare providers for medical advice.
        
        ---
        **Developed by:** AI Healthcare Research Team
        **Version:** 1.0.0
        """)
        
        st.info("💡 To get started, navigate to the **Prediction** mode and upload a chest X-ray image!")


if __name__ == "__main__":
    main()
