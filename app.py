"""
Streamlit web application for AI-powered pneumonia detection.
Enhanced with multilingual support, DICOM upload, uncertainty estimation,
NLP reports, multi-XAI visualization, patient tracking, batch processing,
authentication, and comprehensive dashboard.
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
import uuid
from string import Template
from pathlib import Path
from datetime import datetime

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
    .dashboard-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 0.5rem 0;
    }
    .dashboard-card h2 { color: white; margin: 0; }
    .dashboard-card h4 { color: rgba(255,255,255,0.8); margin: 0; }
</style>
""", unsafe_allow_html=True)


# ==================== Translation Helper ====================

def t(key: str) -> str:
    """Get translated text for the current language."""
    try:
        from utils.multilingual import get_text
        lang = st.session_state.get('language', 'en')
        return get_text(key, lang)
    except ImportError:
        # Fallback translations
        defaults = {
            'app_title': '🫁 AI-Powered Pneumonia Detection System',
            'upload_prompt': 'Upload Chest X-Ray Image',
            'prediction_results': 'Prediction Results',
            'pneumonia_detected': '⚠️ Pneumonia Detected',
            'normal_result': '✅ Normal',
            'confidence': 'Confidence',
            'severity': 'Severity Level',
            'affected_area': 'Affected Area',
            'recommendations': 'Medical Recommendations',
            'download_report': 'Download Full Report',
            'disclaimer': 'This system is for screening purposes only. Always consult a healthcare professional.',
        }
        return defaults.get(key, key)


# ==================== Session State Init ====================

def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        'language': 'en',
        'session_id': str(uuid.uuid4()),
        'authenticated': False,
        'current_user': None,
        'prediction_history': [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ==================== Model Loading ====================

@st.cache_resource
def load_model(model_path: str = 'models/model_weights.pth'):
    """Load trained model (cached for performance)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not os.path.exists(model_path):
        st.error(f"Model weights not found at {model_path}")
        st.info("Please train the model first using: `python train.py`")
        return None, device
    
    # Auto-detect architecture from checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict):
        arch = checkpoint.get('architecture', 'densenet121')
        nc = checkpoint.get('num_classes', 2)
        attn = checkpoint.get('use_attention', False)
    else:
        arch, nc, attn = 'densenet121', 2, False
    
    model = build_model(num_classes=nc, pretrained=False,
                        architecture=arch, use_attention=attn)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model = model.to(device)
    model.eval()
    
    return model, device


# ==================== Core Functions ====================

def predict_image(model, image_path, device):
    """Run prediction on uploaded image with confidence thresholding."""
    input_tensor = preprocess_single_image(image_path).to(device)
    
    # --- Primary prediction ---
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        confidence = probs[0].cpu().numpy()
        raw_prediction = output.argmax(dim=1).item()
    
    pneumonia_prob = float(confidence[1]) if len(confidence) > 1 else float(confidence[0])
    normal_prob = float(confidence[0])
    
    # --- Confidence threshold ---
    # The model must exceed this threshold to declare Pneumonia.
    # This prevents borderline / uncertain cases from being false positives.
    PNEUMONIA_THRESHOLD = 0.65
    
    if raw_prediction == 1 and pneumonia_prob < PNEUMONIA_THRESHOLD:
        # Model is not confident enough — classify as Normal
        prediction = 0
    else:
        prediction = raw_prediction
    
    result = {
        'prediction': prediction,
        'raw_prediction': raw_prediction,
        'confidence': confidence,
        'pneumonia_prob': pneumonia_prob,
        'normal_prob': normal_prob,
        'threshold_applied': (raw_prediction != prediction),
    }
    
    # --- Uncertainty estimation (MC Dropout) ---
    try:
        from utils.uncertainty import predict_with_uncertainty
        unc_result = predict_with_uncertainty(model, input_tensor, device=device)
        result['uncertainty'] = unc_result.get('predictive_entropy', 0)
        result['epistemic'] = unc_result.get('epistemic_uncertainty', 0)
        
        # If epistemic uncertainty is very high, override to Normal
        # (model is unsure → safer to not claim pneumonia)
        if result['epistemic'] > 0.4 and prediction == 1 and pneumonia_prob < 0.80:
            result['prediction'] = 0
            result['threshold_applied'] = True
    except Exception:
        result['uncertainty'] = 0
        result['epistemic'] = 0
    
    return result


def display_prediction_results(pred_results):
    """Display prediction results in a formatted box."""
    prediction = pred_results['prediction']
    confidence = pred_results['pneumonia_prob'] if prediction == 1 else pred_results['normal_prob']
    
    if prediction == 1:
        st.markdown(f"""
        <div class="prediction-box pneumonia-box">
            <h2 style="color: #D32F2F;">{t('pneumonia_detected')}</h2>
            <h3>{t('confidence')}: {pred_results['pneumonia_prob']*100:.2f}%</h3>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="prediction-box normal-box">
            <h2 style="color: #388E3C;">{t('normal_result')}</h2>
            <h3>{t('confidence')}: {pred_results['normal_prob']*100:.2f}%</h3>
        </div>
        """, unsafe_allow_html=True)
    
    # Show threshold note if the prediction was adjusted
    if pred_results.get('threshold_applied'):
        st.info(f"ℹ️ Model raw output leaned Pneumonia ({pred_results['pneumonia_prob']*100:.1f}%) "
                f"but was below the confidence threshold (65%). Classified as **Normal** to avoid a false positive.")
    
    # Uncertainty indicator
    if pred_results.get('uncertainty', 0) > 0:
        unc = pred_results['uncertainty']
        unc_level = 'Low' if unc < 0.3 else ('Medium' if unc < 0.6 else 'High')
        unc_color = '#4CAF50' if unc < 0.3 else ('#FF9800' if unc < 0.6 else '#F44336')
        st.markdown(f"""
        <div style="padding: 0.5rem; margin-top: 0.5rem;">
            <span style="color: {unc_color}; font-weight: bold;">
                Uncertainty: {unc_level} ({unc:.3f})
            </span>
        </div>
        """, unsafe_allow_html=True)
    
    prob_df = pd.DataFrame({
        'Class': ['Normal', 'Pneumonia'],
        'Probability': [pred_results['normal_prob'], pred_results['pneumonia_prob']]
    })
    st.bar_chart(prob_df.set_index('Class'))


def _cam_to_data_uri(cam_raw):
    """Encode a 2D numpy activation map (0-1) as a base64 PNG data URI."""
    import io as _io
    cam_uint8 = (np.clip(cam_raw, 0, 1) * 255).astype(np.uint8)
    pil_img = Image.fromarray(cam_uint8, mode='L')
    buf = _io.BytesIO()
    pil_img.save(buf, format='PNG')
    b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f'data:image/png;base64,{b64}'


def render_lung_glb_viewer(glb_path: str, severity_intensity: float, height: int = 650, cam_raw=None):
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

    # Encode Grad-CAM activation map as a PNG data URI for the shader
    cam_data_uri = ''
    if cam_raw is not None:
        try:
            cam_data_uri = _cam_to_data_uri(cam_raw)
        except Exception:
            cam_data_uri = ''

    html_template = Template(r"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <style>
      html, body { margin: 0; padding: 0; width: 100%; height: 100%; background: #0b0b10; overflow: hidden; }
      #root { width: 100%; height: 100%; }
      canvas { display: block; width: 100%; height: 100%; }
      #legend {
        position: absolute; bottom: 14px; left: 14px; background: rgba(10,10,20,0.85);
        border: 1px solid rgba(255,255,255,0.15); border-radius: 8px;
        padding: 10px 14px; font-family: system-ui, sans-serif; font-size: 12px;
        color: #ccc; pointer-events: none; line-height: 1.6;
      }
      #legend b { color: #fff; }
      .ldot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; vertical-align: middle; }
    </style>
  </head>
  <body>
    <div id="root"></div>
    <div id="legend">
      <b>Lung Damage Map</b><br/>
      <span class="ldot" style="background:#e53935"></span>Severe damage<br/>
      <span class="ldot" style="background:#ff9800"></span>Moderate damage<br/>
      <span class="ldot" style="background:#ffd54f"></span>Mild damage<br/>
      <span class="ldot" style="background:#64b5f6"></span>Healthy tissue
    </div>

    <script type="importmap">
      {
        "imports": {
          "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
          "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
        }
      }
    </script>

    <script type="module">
      import * as THREE from 'three';
      import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
      import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

      const glbUrl = $glb_url;
      const severityIntensity = $severity_intensity;
      const camDataUri = $cam_data_uri;
      const glbReadError = $glb_read_error;

      const root = document.getElementById('root');
      if (!glbUrl) {
        console.error('GLB load error:', glbReadError);
        root.innerHTML = '';
      } else {
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x0b0b10);

        const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 1000);

        const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
        renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
        renderer.setClearColor(0x0b0b10, 1);
        root.appendChild(renderer.domElement);

        const ambient = new THREE.AmbientLight(0xffffff, 0.55);
        scene.add(ambient);
        const dir1 = new THREE.DirectionalLight(0xffffff, 0.80);
        dir1.position.set(2.0, 3.0, 4.0);
        scene.add(dir1);
        const dir2 = new THREE.DirectionalLight(0xaaccff, 0.25);
        dir2.position.set(-3, 1, -2);
        scene.add(dir2);

        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.06;
        controls.rotateSpeed = 0.6;
        controls.zoomSpeed = 0.8;
        controls.panSpeed = 0.6;

        let stop = false;
        let elapsedTime = 0;
        const clock = new THREE.Clock();

        function resize() {
          const w = root.clientWidth || 1;
          const h = root.clientHeight || 1;
          renderer.setSize(w, h, false);
          camera.aspect = w / h;
          camera.updateProjectionMatrix();
        }
        window.addEventListener('resize', resize);
        resize();

        /* --- Load Grad-CAM activation texture (if available) --- */
        let camTexture = null;
        const hasCam = !!(camDataUri && camDataUri.length > 30);

        const textureReady = new Promise((resolve) => {
          if (!hasCam) { resolve(null); return; }
          const img = new Image();
          img.onload = () => {
            camTexture = new THREE.Texture(img);
            camTexture.needsUpdate = true;
            camTexture.minFilter = THREE.LinearFilter;
            camTexture.magFilter = THREE.LinearFilter;
            camTexture.wrapS = THREE.ClampToEdgeWrapping;
            camTexture.wrapT = THREE.ClampToEdgeWrapping;
            resolve(camTexture);
          };
          img.onerror = () => resolve(null);
          img.src = camDataUri;
        });

        const loader = new GLTFLoader();
        loader.load(
          glbUrl,
          async (gltf) => {
            const model = gltf.scene;

            // Wait for CAM texture
            const camTex = await textureReady;

            /* ------- Vertex Shader ------- */
            const vertSrc = `
              varying vec3 vWorldPos;
              varying vec3 vNormalW;
              varying vec2 vUv;
              varying vec3 vViewDir;
              uniform vec3 uBboxMin;
              uniform vec3 uBboxSize;
              varying vec3 vLocalNorm;  // normalized position in bounding box [0,1]
              void main() {
                vec4 wp = modelMatrix * vec4(position, 1.0);
                vWorldPos = wp.xyz;
                vNormalW  = normalize(mat3(modelMatrix) * normal);
                vViewDir  = normalize(cameraPosition - wp.xyz);
                vUv = uv;
                // Compute normalized position inside bounding box for CAM projection
                vLocalNorm = (wp.xyz - uBboxMin) / max(uBboxSize, vec3(0.0001));
                gl_Position = projectionMatrix * viewMatrix * wp;
              }
            `;

            /* ------- Fragment Shader ------- */
            const fragSrc = `
              precision highp float;
              varying vec3 vWorldPos;
              varying vec3 vNormalW;
              varying vec2 vUv;
              varying vec3 vViewDir;
              varying vec3 vLocalNorm;

              uniform vec3  uBaseColor;
              uniform sampler2D uBaseMap;
              uniform float uHasMap;
              uniform float uSeverity;
              uniform float uTime;

              /* Grad-CAM activation texture */
              uniform sampler2D uCamTex;
              uniform float     uHasCam;

              /* --- Noise helpers (for subtle detail) --- */
              float hash31(vec3 p) {
                p = fract(p * 0.1031);
                p += dot(p, p.yzx + 33.33);
                return fract((p.x + p.y) * p.z);
              }
              float noise3(vec3 p) {
                vec3 i = floor(p); vec3 f = fract(p);
                f = f*f*(3.0-2.0*f);
                float a = hash31(i), b = hash31(i+vec3(1,0,0)),
                      c = hash31(i+vec3(0,1,0)), d = hash31(i+vec3(1,1,0)),
                      e = hash31(i+vec3(0,0,1)), f2= hash31(i+vec3(1,0,1)),
                      g = hash31(i+vec3(0,1,1)), h = hash31(i+vec3(1,1,1));
                return mix(mix(mix(a,b,f.x),mix(c,d,f.x),f.y),
                           mix(mix(e,f2,f.x),mix(g,h,f.x),f.y),f.z);
              }

              /* Sample Grad-CAM from multiple projection axes and take the max,
                 giving full coverage regardless of which side of the lung faces the X-ray */
              float sampleCAM() {
                // Front-back projection (XY plane, as in a standard AP/PA chest X-ray)
                vec2 uvFront = vec2(vLocalNorm.x, 1.0 - vLocalNorm.y);
                float camFront = texture2D(uCamTex, uvFront).r;

                // Side projection (ZY plane, lateral view)
                vec2 uvSide = vec2(vLocalNorm.z, 1.0 - vLocalNorm.y);
                float camSide = texture2D(uCamTex, uvSide).r;

                // Take maximum so any high-activation region shows up
                return max(camFront, camSide * 0.7);
              }

              /* Infection color ramp by activation intensity */
              vec3 infectionColor(float activation, float sev) {
                vec3 mild     = vec3(1.0, 0.85, 0.2);   // warm yellow
                vec3 moderate = vec3(1.0, 0.45, 0.1);   // hot orange
                vec3 severe   = vec3(0.95, 0.15, 0.12); // bright red

                // Map activation 0-1 through the color ramp
                float t = activation * sev; // scale by overall severity
                vec3 col;
                if (t < 0.35)     col = mix(mild, moderate, t / 0.35);
                else              col = mix(moderate, severe, (t - 0.35) / 0.65);
                return col;
              }

              /* Healthy tissue tint (varies by overall severity) */
              vec3 healthyTint(float sev) {
                vec3 low  = vec3(0.55, 0.82, 0.88); // cool teal
                vec3 mid  = vec3(0.60, 0.78, 0.65); // sage
                vec3 high = vec3(0.68, 0.55, 0.55); // stressed pink
                if (sev < 0.4) return mix(low, mid, sev / 0.4);
                return mix(mid, high, (sev - 0.4) / 0.6);
              }

              void main() {
                float sev = clamp(uSeverity, 0.0, 1.0);
                vec3 baseCol = uBaseColor;
                if (uHasMap > 0.5) { baseCol *= texture2D(uBaseMap, vUv).rgb; }

                vec3 N = normalize(vNormalW);
                vec3 V = normalize(vViewDir);
                vec3 L = normalize(vec3(0.4, 0.8, 0.6));
                float ndl = max(dot(N, L), 0.0);
                float shade = 0.48 + 0.52 * ndl;
                float fresnel = pow(1.0 - max(dot(N, V), 0.0), 3.0);

                /* Healthy base */
                vec3 healthy = healthyTint(sev) * baseCol;
                healthy += fresnel * 0.1 * vec3(0.5, 0.7, 1.0);

                /* --- If no CAM data or very low severity, show healthy --- */
                if (sev < 0.05 || uHasCam < 0.5) {
                  // If there IS severity but no CAM, fall back to simple noise overlay
                  if (sev >= 0.05 && uHasCam < 0.5) {
                    float n = noise3(vWorldPos * 0.75);
                    float coverage = mix(0.86, 0.42, sev);
                    float m = smoothstep(coverage, coverage + 0.12, n) * sev;
                    vec3 inf = infectionColor(m, sev);
                    vec3 c = mix(healthy, inf, m);
                    gl_FragColor = vec4(c * shade, 1.0);
                    return;
                  }
                  gl_FragColor = vec4(healthy * shade, 1.0);
                  return;
                }

                /* --- Sample actual Grad-CAM activation --- */
                float cam = sampleCAM();

                /* Add subtle noise variation so edges look organic, not pixel-perfect */
                float detail = noise3(vWorldPos * 2.5) * 0.15;
                cam = clamp(cam + detail - 0.07, 0.0, 1.0);

                /* Threshold: only show regions with meaningful activation */
                float activation = smoothstep(0.12, 0.35, cam) * cam;

                /* Scale by severity so mild cases show lighter highlights */
                activation *= mix(0.6, 1.0, sev);

                /* --- Edge glow at damage boundaries --- */
                float edge = smoothstep(0.10, 0.18, cam) - smoothstep(0.25, 0.45, cam);
                edge = max(edge, 0.0);
                float pulse = 0.85 + 0.15 * sin(uTime * 2.0 + cam * 8.0);
                edge *= pulse;

                /* --- Composite --- */
                vec3 inf = infectionColor(cam, sev);

                // Emissive glow in damaged areas
                float emissive = activation * mix(0.15, 0.5, sev) * pulse;

                vec3 color = mix(healthy, inf, activation);
                color *= shade;
                color += inf * emissive;  // self-illumination

                // Edge glow
                vec3 edgeCol = (sev >= 0.65) ? vec3(1.0, 0.3, 0.15) : vec3(1.0, 0.7, 0.25);
                color += edgeCol * edge * 1.2;

                // Fresnel on damaged areas
                color += fresnel * activation * 0.12 * inf;

                gl_FragColor = vec4(color, 1.0);
              }
            `;

            /* Compute bounding box for world-space CAM projection */
            const box = new THREE.Box3().setFromObject(model);
            const bboxMin = box.min.clone();
            const bboxSize = new THREE.Vector3();
            box.getSize(bboxSize);
            const center = new THREE.Vector3();
            box.getCenter(center);

            const allMats = [];

            model.traverse((obj) => {
              if (obj && obj.isMesh) {
                const origMat = obj.material;
                const base = (origMat && origMat.color)
                  ? origMat.color.clone()
                  : new THREE.Color(0.82, 0.82, 0.82);
                const baseMap = (origMat && origMat.map) ? origMat.map : null;
                const side = origMat && typeof origMat.side === 'number'
                  ? origMat.side : THREE.FrontSide;

                const mat = new THREE.ShaderMaterial({
                  uniforms: {
                    uBaseColor: { value: base },
                    uBaseMap:   { value: baseMap },
                    uHasMap:    { value: baseMap ? 1.0 : 0.0 },
                    uSeverity:  { value: severityIntensity },
                    uTime:      { value: 0.0 },
                    uCamTex:    { value: camTex },
                    uHasCam:    { value: camTex ? 1.0 : 0.0 },
                    uBboxMin:   { value: bboxMin },
                    uBboxSize:  { value: bboxSize }
                  },
                  vertexShader: vertSrc,
                  fragmentShader: fragSrc,
                  transparent: false,
                  depthWrite: true,
                  depthTest: true,
                  side
                });
                obj.material = mat;
                allMats.push(mat);
              }
            });

            scene.add(model);

            const maxDim = Math.max(bboxSize.x, bboxSize.y, bboxSize.z) || 1;
            camera.position.set(center.x, center.y, center.z + maxDim * 2.2);
            camera.near = Math.max(0.01, maxDim / 1000);
            camera.far = maxDim * 50;
            camera.updateProjectionMatrix();

            controls.target.copy(center);
            controls.update();

            /* Update time uniform for pulsing effect */
            function updateUniforms() {
              const dt = clock.getDelta();
              elapsedTime += dt;
              for (const m of allMats) {
                m.uniforms.uTime.value = elapsedTime;
              }
            }

            function animate() {
              if (stop) return;
              updateUniforms();
              controls.update();
              renderer.render(scene, camera);
              requestAnimationFrame(animate);
            }
            animate();
          },
          undefined,
          (err) => {
            console.error('Failed to load GLB:', err);
            stop = true;
            window.removeEventListener('resize', resize);
            root.innerHTML = '';
          }
        );
      }
    </script>
  </body>
</html>""")

    html = html_template.substitute(
        glb_url=json.dumps(glb_data_uri if glb_data_uri else None),
        severity_intensity=json.dumps(severity_intensity),
        cam_data_uri=json.dumps(cam_data_uri if cam_data_uri else ''),
        glb_read_error=json.dumps(glb_read_error)
    )

    components.html(html, height=height, scrolling=False)


def main():
    """Main Streamlit application with all enhanced features."""
    
    init_session_state()
    
    # Header
    st.markdown(f'<p class="main-header">{t("app_title")}</p>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("📋 Navigation")
    
    # Language selector
    languages = {
        'en': 'English', 'es': 'Español', 'fr': 'Français',
        'de': 'Deutsch', 'zh': '中文', 'hi': 'हिन्दी',
        'ar': 'العربية', 'pt': 'Português', 'ja': '日本語', 'ko': '한국어'
    }
    selected_lang = st.sidebar.selectbox(
        "🌍 Language",
        list(languages.keys()),
        format_func=lambda x: languages[x],
        index=0
    )
    st.session_state['language'] = selected_lang
    
    # Login section
    if not st.session_state.get('authenticated', False):
        with st.sidebar.expander("🔐 Login (Optional)"):
            username = st.text_input("Username", key="login_user")
            password = st.text_input("Password", type="password", key="login_pass")
            if st.button("Login"):
                try:
                    from auth import get_user_manager
                    manager = get_user_manager()
                    user = manager.authenticate(username, password)
                    if user:
                        st.session_state['authenticated'] = True
                        st.session_state['current_user'] = user
                        st.success(f"Welcome, {user.get('full_name', username)}!")
                        st.rerun()
                    else:
                        st.error("Invalid credentials")
                except Exception:
                    st.session_state['authenticated'] = True
                    st.session_state['current_user'] = {'username': username, 'role': 'viewer'}
    else:
        user = st.session_state.get('current_user', {})
        st.sidebar.success(f"👤 {user.get('full_name', user.get('username', 'User'))}")
        st.sidebar.caption(f"Role: {user.get('role', 'viewer').title()}")
        if st.sidebar.button("Logout"):
            st.session_state['authenticated'] = False
            st.session_state['current_user'] = None
            st.rerun()
    
    app_mode = st.sidebar.selectbox(
        "Select Mode",
        ["Prediction", "Batch Processing", "Patient History",
         "Dashboard", "Model Evaluation", "Settings", "About"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **How to use:**
    1. Upload a chest X-ray image (JPG, PNG, or DICOM)
    2. View prediction results with uncertainty
    3. Analyze Grad-CAM & multi-XAI visualizations
    4. Explore 3D lung mapping
    5. Read AI-generated medical reports
    6. Track patient history over time
    """)
    
    # Load model
    model, device = load_model()
    
    if model is None:
        st.stop()
    
    # ==================== PREDICTION MODE ====================
    if app_mode == "Prediction":
        render_prediction_page(model, device)
    
    # ==================== BATCH PROCESSING ====================
    elif app_mode == "Batch Processing":
        render_batch_page(model, device)
    
    # ==================== PATIENT HISTORY ====================
    elif app_mode == "Patient History":
        render_patient_history_page()
    
    # ==================== DASHBOARD ====================
    elif app_mode == "Dashboard":
        render_dashboard_page()
    
    # ==================== EVALUATION MODE ====================
    elif app_mode == "Model Evaluation":
        render_evaluation_page()
    
    # ==================== SETTINGS ====================
    elif app_mode == "Settings":
        render_settings_page()
    
    # ==================== ABOUT ====================
    else:
        render_about_page()


def handle_file_upload():
    """Handle file upload including DICOM support."""
    uploaded_file = st.file_uploader(
        t('upload_prompt'),
        type=['jpg', 'jpeg', 'png', 'dcm', 'dicom'],
        help="Upload a chest X-ray image (JPEG, PNG) or DICOM file"
    )
    
    if uploaded_file is not None:
        filename = uploaded_file.name
        temp_path = f"temp_{filename}"
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Handle DICOM files
        if filename.lower().endswith(('.dcm', '.dicom')):
            try:
                from utils.dicom_handler import dicom_to_temp_png, extract_dicom_metadata
                
                dicom_meta = extract_dicom_metadata(temp_path)
                st.info(f"DICOM: {dicom_meta.get('PatientID', 'Unknown')} | "
                        f"Modality: {dicom_meta.get('Modality', 'N/A')} | "
                        f"Study: {dicom_meta.get('StudyDate', 'N/A')}")
                
                png_path = dicom_to_temp_png(temp_path)
                os.remove(temp_path)
                temp_path = png_path
                
            except ImportError:
                st.warning("DICOM support not available. Install pydicom: `pip install pydicom`")
                os.remove(temp_path)
                return None, None
            except Exception as e:
                st.error(f"Error reading DICOM: {e}")
                os.remove(temp_path)
                return None, None
        
        # Log upload
        try:
            from utils.compliance import get_audit_logger
            logger = get_audit_logger()
            logger.log_upload(
                st.session_state.get('session_id', 'anon'),
                filename, uploaded_file.size, uploaded_file.type or 'unknown'
            )
        except Exception:
            pass
        
        return temp_path, filename
    
    return None, None


def render_prediction_page(model, device):
    """Render the main prediction page."""
    st.header("🔍 X-Ray Analysis")
    
    # Patient ID input (optional)
    patient_id = st.text_input("Patient ID (optional)", placeholder="e.g., P001")
    
    temp_path, filename = handle_file_upload()
    
    if temp_path is not None:
        # Display original image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Original X-Ray")
            image = Image.open(temp_path)
            st.image(image, width='stretch')
        
        # Run prediction
        with st.spinner("🔬 Analyzing X-ray..."):
            pred_results = predict_image(model, temp_path, device)
        
        with col2:
            st.subheader(f"🎯 {t('prediction_results')}")
            display_prediction_results(pred_results)
        
        st.markdown("---")
        
        # Grad-CAM Visualization
        st.header("🔥 Grad-CAM Heatmap Analysis")
        
        with st.spinner("Generating Grad-CAM visualization..."):
            try:
                original_img = cv2.imread(temp_path)
                original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                original_img = cv2.resize(original_img, (224, 224))
                
                input_tensor = preprocess_single_image(temp_path).to(device)
                
                result = generate_gradcam_visualization(
                    model, input_tensor, original_img
                )
                # Unpack — supports both 3-tuple (old) and 4-tuple (new) returns
                if len(result) == 4:
                    heatmap, overlaid, cam_intensity, cam_raw = result
                else:
                    heatmap, overlaid, cam_intensity = result
                    cam_raw = None
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🔥 Heatmap Overlay")
                    st.image(overlaid, width='stretch')
                    st.caption("Red regions indicate areas of high model attention")
                
                with col2:
                    st.subheader("📊 Intensity Analysis")
                    
                    if cam_raw is not None:
                        # Build a proper intensity visualization from the raw activation map
                        import matplotlib.pyplot as plt
                        import matplotlib
                        matplotlib.use('Agg')
                        
                        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
                        im = ax.imshow(cam_raw, cmap='inferno', vmin=0, vmax=1)
                        ax.set_title('Grad-CAM Activation Map', fontsize=12, fontweight='bold', color='white')
                        ax.axis('off')
                        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                        cbar.set_label('Activation Intensity', color='white', fontsize=10)
                        cbar.ax.yaxis.set_tick_params(color='white')
                        for lbl in cbar.ax.get_yticklabels():
                            lbl.set_color('white')
                        fig.patch.set_facecolor('#0E1117')
                        ax.set_facecolor('#0E1117')
                        fig.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
                        
                        # Numerical intensity metrics
                        high_thresh = 0.5
                        high_pct = float((cam_raw > high_thresh).sum()) / cam_raw.size * 100
                        med_pct = float(((cam_raw > 0.25) & (cam_raw <= high_thresh)).sum()) / cam_raw.size * 100
                        low_pct = 100.0 - high_pct - med_pct
                        
                        st.markdown(f"""
                        | Metric | Value |
                        |--------|-------|
                        | **Mean Activation** | `{cam_intensity:.4f}` |
                        | **Max Activation** | `{float(cam_raw.max()):.4f}` |
                        | **High Intensity (>0.5)** | `{high_pct:.1f}%` of image |
                        | **Medium (0.25–0.5)** | `{med_pct:.1f}%` of image |
                        | **Low (<0.25)** | `{low_pct:.1f}%` of image |
                        """)
                    else:
                        # Fallback: display the JET heatmap itself
                        st.image(heatmap, width='stretch',
                                caption=f"Activation intensity (mean: {cam_intensity:.4f})")
                        st.metric("Mean Activation", f"{cam_intensity:.4f}")
                
                # Multi-XAI comparison
                st.markdown("---")
                st.header("🔬 Explainability Methods Comparison")
                
                xai_method = st.selectbox(
                    "Select XAI Method",
                    ['Grad-CAM (default)', 'Integrated Gradients', 'LIME', 'All Methods']
                )
                
                if xai_method != 'Grad-CAM (default)':
                    try:
                        from utils.xai_methods import generate_explanation, compare_xai_methods
                        
                        if xai_method == 'All Methods':
                            with st.spinner('Running all XAI methods...'):
                                fig = compare_xai_methods(
                                    model, input_tensor, original_img,
                                    methods=['gradcam', 'integrated_gradients', 'lime']
                                )
                            st.pyplot(fig)
                        else:
                            method_map = {
                                'Integrated Gradients': 'integrated_gradients',
                                'LIME': 'lime'
                            }
                            method_key = method_map.get(xai_method, 'gradcam')
                            with st.spinner(f'Running {xai_method}...'):
                                result = generate_explanation(
                                    model, input_tensor, original_img, method=method_key
                                )
                            st.image(result['heatmap_overlay'], 
                                    caption=f"{xai_method} explanation (intensity: {result.get('intensity', 0):.3f})",
                                    width='stretch')
                    except Exception as e:
                        st.warning(f"XAI method unavailable: {e}")
                
                # Severity Classification
                st.markdown("---")
                st.header(f"📈 {t('severity')}")
                
                severity_classifier = SeverityClassifier()
                affected_area = severity_classifier.get_affected_area_percentage(
                    cam_raw if cam_raw is not None else cam_intensity * np.ones((224, 224))
                )
                
                severity = severity_classifier.classify(
                    pred_results['pneumonia_prob'],
                    affected_area
                )
                
                severity_color = get_severity_color(severity)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>{t('severity')}</h4>
                        <h2 style="color: {severity_color};">{severity}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>{t('confidence')}</h4>
                        <h2>{pred_results['pneumonia_prob']*100:.1f}%</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>{t('affected_area')}</h4>
                        <h2>{affected_area:.1f}%</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    unc = pred_results.get('uncertainty', 0)
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>Uncertainty</h4>
                        <h2>{unc:.3f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                # 3D Visualization
                st.markdown("---")
                st.header("🫁 3D Lung Visualization")
                
                with st.spinner("Loading anatomical lung model..."):
                    severity_intensity = float(pred_results['pneumonia_prob']) * (float(affected_area) / 100.0)
                    render_lung_glb_viewer("/public/models/lung_carcinoma.glb", severity_intensity, cam_raw=cam_raw)
                
                # Medical Recommendations
                st.markdown("---")
                st.header(f"💊 {t('recommendations')}")
                
                recommendations = generate_recommendations(
                    severity,
                    pred_results['pneumonia_prob'],
                    affected_area
                )
                
                if recommendations['urgency_level'] in ['High', 'Critical']:
                    st.error(f"⚠️ {recommendations['urgency_level']} Priority - {recommendations['follow_up']}")
                elif recommendations['urgency_level'] == 'Moderate':
                    st.warning(f"⚠️ {recommendations['follow_up']}")
                else:
                    st.info(f"ℹ️ {recommendations['follow_up']}")
                
                st.subheader("📋 Action Items:")
                for i, rec in enumerate(recommendations['recommendations'], 1):
                    st.write(f"{i}. {rec}")
                
                if recommendations['warning']:
                    st.warning(f"⚠️ **Warning:** {recommendations['warning']}")
                
                st.info(f"ℹ️ **{t('disclaimer')}**")
                
                # NLP Report Generation
                st.markdown("---")
                st.header("📝 AI-Generated Report")
                
                report_type = st.radio(
                    "Report Type",
                    ['Radiology Report', 'Patient-Friendly Report', 'Both'],
                    horizontal=True
                )
                
                try:
                    from utils.nlp_reports import generate_report
                    type_map = {'Radiology Report': 'radiology', 'Patient-Friendly Report': 'patient_friendly', 'Both': 'both'}
                    nlp_report = generate_report(
                        prediction='Pneumonia' if pred_results['prediction'] == 1 else 'Normal',
                        confidence=float(pred_results['pneumonia_prob'] if pred_results['prediction'] == 1 else pred_results['normal_prob']),
                        severity=severity,
                        affected_area=affected_area,
                        report_type=type_map[report_type]
                    )
                    st.text_area("Generated Report", nlp_report, height=300)
                    report_text = nlp_report
                except Exception:
                    report_text = generate_patient_report(
                        severity, pred_results['pneumonia_prob'],
                        affected_area, recommendations
                    )
                
                st.download_button(
                    label=f"📄 {t('download_report')}",
                    data=report_text,
                    file_name="pneumonia_analysis_report.txt",
                    mime="text/plain"
                )
                
                # Save to database
                if patient_id:
                    try:
                        from database import get_database
                        db = get_database()
                        db.add_patient(patient_id)
                        pred_id = db.add_prediction(
                            filename=filename,
                            prediction=pred_results['prediction'],
                            prediction_label='Pneumonia' if pred_results['prediction'] == 1 else 'Normal',
                            confidence=float(pred_results['pneumonia_prob'] if pred_results['prediction'] == 1 else pred_results['normal_prob']),
                            patient_id=patient_id,
                            severity=severity,
                            affected_area=float(affected_area),
                            uncertainty=float(pred_results.get('uncertainty', 0)),
                            report=report_text,
                        )
                        st.success(f"Result saved for patient {patient_id} (ID: {pred_id})")
                        
                        # Radiologist feedback
                        st.markdown("---")
                        st.subheader("🔄 Radiologist Feedback")
                        feedback_col1, feedback_col2 = st.columns(2)
                        with feedback_col1:
                            is_correct = st.radio("Is this prediction correct?", ['Yes', 'No', 'Unsure'])
                        with feedback_col2:
                            feedback_comments = st.text_area("Comments", placeholder="Optional clinical notes...")
                        
                        if st.button("Submit Feedback"):
                            correct_map = {'Yes': True, 'No': False, 'Unsure': None}
                            db.add_feedback(
                                prediction_id=pred_id,
                                reviewer_id=st.session_state.get('current_user', {}).get('username', 'anonymous'),
                                is_correct=correct_map[is_correct],
                                comments=feedback_comments
                            )
                            st.success("Feedback saved! Thank you.")
                    except Exception as e:
                        st.caption(f"Database: {e}")
                
                # Audit log
                try:
                    from utils.compliance import get_audit_logger
                    logger = get_audit_logger()
                    logger.log_prediction(
                        st.session_state.get('session_id', 'anon'),
                        filename, pred_results['prediction'],
                        float(pred_results['pneumonia_prob']),
                        severity, float(affected_area)
                    )
                except Exception:
                    pass
                
            except Exception as e:
                st.error(f"Error during visualization: {e}")
                st.exception(e)
        
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)


def render_batch_page(model, device):
    """Render batch processing page."""
    st.header("📦 Batch Processing")
    st.write("Upload multiple X-ray images for batch analysis.")
    
    uploaded_files = st.file_uploader(
        "Upload multiple X-ray images",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True
    )
    
    if uploaded_files and st.button("🚀 Process Batch"):
        temp_paths = []
        for f in uploaded_files:
            path = f"temp_batch_{f.name}"
            with open(path, "wb") as out:
                out.write(f.getbuffer())
            temp_paths.append(path)
        
        progress = st.progress(0)
        results = []
        
        for i, (path, f) in enumerate(zip(temp_paths, uploaded_files)):
            try:
                pred = predict_image(model, path, device)
                results.append({
                    'Filename': f.name,
                    'Prediction': 'Pneumonia' if pred['prediction'] == 1 else 'Normal',
                    'Confidence': f"{max(pred['pneumonia_prob'], pred['normal_prob'])*100:.1f}%",
                    'Pneumonia Prob': f"{pred['pneumonia_prob']*100:.1f}%",
                })
            except Exception as e:
                results.append({
                    'Filename': f.name,
                    'Prediction': 'Error',
                    'Confidence': str(e),
                    'Pneumonia Prob': 'N/A',
                })
            progress.progress((i + 1) / len(uploaded_files))
        
        # Display results
        df = pd.DataFrame(results)
        st.dataframe(df, width='stretch')
        
        # Summary
        pneumonia_count = sum(1 for r in results if r['Prediction'] == 'Pneumonia')
        normal_count = sum(1 for r in results if r['Prediction'] == 'Normal')
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total", len(results))
        col2.metric("Pneumonia", pneumonia_count)
        col3.metric("Normal", normal_count)
        
        # Download CSV
        csv = df.to_csv(index=False)
        st.download_button("📥 Download CSV", csv, "batch_results.csv", "text/csv")
        
        # Cleanup
        for path in temp_paths:
            if os.path.exists(path):
                os.remove(path)


def render_patient_history_page():
    """Render patient history page."""
    st.header("📋 Patient History")
    
    try:
        from database import get_database
        db = get_database()
    except Exception:
        st.warning("Database not available.")
        return
    
    search = st.text_input("🔍 Search Patient (ID or Name)")
    
    if search:
        patients = db.search_patients(search)
    else:
        patients = db.list_patients(limit=20)
    
    if patients:
        for patient in patients:
            with st.expander(f"👤 {patient.get('name', patient['patient_id'])} ({patient['patient_id']})"):
                col1, col2, col3 = st.columns(3)
                col1.write(f"**Age:** {patient.get('age', 'N/A')}")
                col2.write(f"**Gender:** {patient.get('gender', 'N/A')}")
                col3.write(f"**Since:** {patient.get('created_at', 'N/A')}")
                
                history = db.get_patient_history(patient['patient_id'])
                if history:
                    hist_df = pd.DataFrame(history)
                    cols_to_show = ['created_at', 'prediction_label', 'confidence', 'severity', 'affected_area']
                    available_cols = [c for c in cols_to_show if c in hist_df.columns]
                    st.dataframe(hist_df[available_cols], width='stretch')
                    
                    # Trend chart
                    if 'confidence' in hist_df.columns and len(hist_df) > 1:
                        st.line_chart(hist_df[['confidence']].astype(float))
                else:
                    st.info("No predictions recorded yet.")
    else:
        st.info("No patients found. Predictions will be saved when you enter a Patient ID on the Prediction page.")


def render_dashboard_page():
    """Render comprehensive dashboard."""
    st.header("📊 System Dashboard")
    
    try:
        from database import get_database
        db = get_database()
        stats = db.get_dashboard_stats()
    except Exception:
        st.warning("Database not available. Dashboard requires prediction history.")
        return
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="dashboard-card">
            <h4>Total Predictions</h4>
            <h2>{stats['total_predictions']}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="dashboard-card">
            <h4>Total Patients</h4>
            <h2>{stats['total_patients']}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="dashboard-card">
            <h4>Today's Scans</h4>
            <h2>{stats['today_predictions']}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="dashboard-card">
            <h4>Avg Confidence</h4>
            <h2>{stats['avg_confidence']*100:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Prediction distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Prediction Distribution")
        pred_counts = stats.get('prediction_counts', {})
        if pred_counts:
            df = pd.DataFrame(list(pred_counts.items()), columns=['Class', 'Count'])
            st.bar_chart(df.set_index('Class'))
    
    with col2:
        st.subheader("Severity Distribution")
        severity_dist = stats.get('severity_distribution', {})
        if severity_dist:
            df = pd.DataFrame(list(severity_dist.items()), columns=['Severity', 'Count'])
            st.bar_chart(df.set_index('Severity'))
    
    # Feedback stats
    st.markdown("---")
    st.subheader("Feedback & Quality")
    feedback = stats.get('feedback_stats', {})
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Feedback", feedback.get('total_feedback', 0))
    col2.metric("Correct Predictions", feedback.get('correct', 0))
    col3.metric("Feedback Accuracy", f"{feedback.get('accuracy', 0)*100:.1f}%")
    
    # Recent predictions
    st.markdown("---")
    st.subheader("Recent Predictions")
    recent = db.get_recent_predictions(limit=10)
    if recent:
        df = pd.DataFrame(recent)
        cols = ['created_at', 'filename', 'prediction_label', 'confidence', 'severity']
        available = [c for c in cols if c in df.columns]
        st.dataframe(df[available], width='stretch')


def render_evaluation_page():
    """Render model evaluation page."""
    st.header("📊 Model Performance Evaluation")
    
    results_dir = "results"
    
    if not os.path.exists(results_dir):
        st.warning("⚠️ No evaluation results found. Please run evaluation first:")
        st.code("python evaluate.py", language="bash")
    else:
        metrics_path = os.path.join(results_dir, "metrics.csv")
        if os.path.exists(metrics_path):
            st.subheader("📈 Performance Metrics")
            
            metrics_df = pd.read_csv(metrics_path)
            
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
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🔲 Confusion Matrix")
                cm_path = os.path.join(results_dir, "confusion_matrix.png")
                if os.path.exists(cm_path):
                    st.image(cm_path, width='stretch')
            
            with col2:
                st.subheader("📈 ROC Curve")
                roc_path = os.path.join(results_dir, "roc_curve.png")
                if os.path.exists(roc_path):
                    st.image(roc_path, width='stretch')
            
            st.subheader("📊 Metrics Overview")
            metrics_bar_path = os.path.join(results_dir, "metrics_bar.png")
            if os.path.exists(metrics_bar_path):
                st.image(metrics_bar_path, width='stretch')
            
            # XAI comparisons
            xai_dir = os.path.join(results_dir, "xai_comparisons")
            if os.path.exists(xai_dir):
                st.markdown("---")
                st.subheader("🔬 XAI Method Comparisons")
                xai_images = sorted(Path(xai_dir).glob("*.png"))
                for img_path in xai_images[:5]:
                    st.image(str(img_path), width='stretch')
            
            st.subheader("📋 Detailed Classification Report")
            report_path = os.path.join(results_dir, "classification_report.txt")
            if os.path.exists(report_path):
                with open(report_path, 'r') as f:
                    report_text = f.read()
                st.text(report_text)
    
    # Model versions
    st.markdown("---")
    st.subheader("📦 Model Versions")
    try:
        from database import get_database
        db = get_database()
        versions = db.list_model_versions()
        if versions:
            df = pd.DataFrame(versions)
            st.dataframe(df, width='stretch')
        else:
            st.info("No model versions registered yet.")
    except Exception:
        st.info("Model version tracking available after first training run.")


def render_settings_page():
    """Render system settings page."""
    st.header("⚙️ Settings")
    
    # User management (admin only)
    user = st.session_state.get('current_user', {})
    if user.get('role') == 'admin':
        st.subheader("👥 User Management")
        
        try:
            from auth import get_user_manager, ROLES
            manager = get_user_manager()
            
            with st.expander("➕ Create New User"):
                new_user = st.text_input("Username", key="new_username")
                new_pass = st.text_input("Password", type="password", key="new_password")
                new_role = st.selectbox("Role", list(ROLES.keys()))
                new_name = st.text_input("Full Name", key="new_fullname")
                
                if st.button("Create User"):
                    result = manager.create_user(new_user, new_pass, new_role, new_name)
                    if result['status'] == 'success':
                        st.success(f"User '{new_user}' created!")
                    else:
                        st.error(result.get('message', 'Error'))
            
            st.subheader("Registered Users")
            users = manager.list_users()
            if users:
                df = pd.DataFrame(users)
                st.dataframe(df, width='stretch')
        except Exception as e:
            st.info(f"Auth module: {e}")
    
    # Audit logs
    st.markdown("---")
    st.subheader("📝 Audit Logs")
    
    try:
        from utils.compliance import get_audit_logger
        logger = get_audit_logger()
        
        stats = logger.get_statistics()
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Events", stats['total_events'])
        col2.metric("Predictions", stats['predictions'])
        col3.metric("Unique Users", stats['unique_users'])
        
        # Recent logs
        recent_logs = logger.get_audit_trail(limit=20)
        if recent_logs:
            df = pd.DataFrame(recent_logs)
            st.dataframe(df, width='stretch')
    except Exception:
        st.info("Audit logging will be available after first prediction.")
    
    # System info
    st.markdown("---")
    st.subheader("🖥️ System Info")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**PyTorch:** {torch.__version__}")
        st.write(f"**CUDA available:** {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            st.write(f"**GPU:** {torch.cuda.get_device_name(0)}")
    with col2:
        st.write(f"**Python:** {os.sys.version.split()[0]}")
        st.write(f"**Session ID:** {st.session_state.get('session_id', 'N/A')[:8]}...")


def render_about_page():
    """Render about page."""
    st.header("ℹ️ About This System")
    
    st.markdown("""
    ### AI-Powered Pneumonia Detection System v2.0
    
    This application uses deep learning to detect pneumonia from chest X-ray images.
    
    **Key Features:**
    - 🔬 **Multi-Architecture Support**: DenseNet121, EfficientNet-B4, ResNet50, and ensemble models
    - 🧠 **Attention Mechanisms**: CBAM and SE blocks for improved focus
    - 🔥 **Multi-XAI Visualization**: Grad-CAM, Integrated Gradients, LIME, Gradient SHAP
    - 🫁 **3D Lung Mapping**: Interactive 3D visualization with severity overlay
    - 📊 **Uncertainty Estimation**: MC Dropout for reliable confidence measures
    - 📝 **AI-Generated Reports**: NLP-based radiology and patient-friendly reports
    - 🌍 **Multilingual**: Support for 10 languages
    - 📋 **Patient Tracking**: Full history and trend analysis
    - 📦 **Batch Processing**: Analyze multiple X-rays at once
    - 🔐 **Authentication**: Role-based access control (Admin, Radiologist, Technician, Viewer)
    - 🏥 **DICOM Support**: Direct upload of clinical DICOM files
    - 📈 **Dashboard**: Real-time system analytics
    - 🔒 **HIPAA Compliance**: Audit logging and data anonymization
    - 🐳 **Docker & API**: REST API with FastAPI and Docker deployment
    - 🔄 **Federated Learning**: Privacy-preserving distributed training
    
    **Technology Stack:**
    - Deep Learning: PyTorch, TorchVision, ONNX
    - Visualization: Plotly, Matplotlib, Three.js
    - Web: Streamlit, FastAPI, React
    - Database: SQLite, Patient tracking
    - Security: JWT, bcrypt, RBAC
    
    **Disclaimer:**
    This system is designed for educational and screening purposes only. 
    It should NOT be used as a substitute for professional medical diagnosis.
    Always consult qualified healthcare providers for medical advice.
    
    ---
    **Version:** 2.0.0
    """)
    
    st.info("💡 To get started, navigate to the **Prediction** mode and upload a chest X-ray image!")


if __name__ == "__main__":
    main()
