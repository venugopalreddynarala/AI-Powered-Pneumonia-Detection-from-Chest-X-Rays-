import React, { useMemo, useState, useEffect } from 'react';
import LungViewer from './LungViewer';
import ErrorBoundary from './ErrorBoundary';
import { normalizeGradcamStat, severityToColor } from './severity';

// API configuration
const API_BASE = 'http://localhost:8000';

interface PredictionResult {
  prediction_id: number;
  filename: string;
  prediction: number;
  prediction_label: string;
  confidence: number;
  severity?: string;
  affected_area?: number;
  uncertainty?: number;
  recommendations?: string[];
}

export default function App() {
  const [gradcamMean, setGradcamMean] = useState(0.35);
  const [animate, setAnimate] = useState(true);
  const [activeTab, setActiveTab] = useState<'viewer' | 'upload' | 'history'>('viewer');
  const [predictionResult, setPredictionResult] = useState<PredictionResult | null>(null);
  const [uploading, setUploading] = useState(false);
  const [apiStatus, setApiStatus] = useState<'checking' | 'online' | 'offline'>('checking');

  const severity = useMemo(() => {
    return normalizeGradcamStat(gradcamMean, 0.0, 1.0);
  }, [gradcamMean]);

  const sevMeta = severityToColor(severity);

  // Check API status
  useEffect(() => {
    fetch(`${API_BASE}/health`)
      .then(r => r.json())
      .then(() => setApiStatus('online'))
      .catch(() => setApiStatus('offline'));
  }, []);

  // Handle file upload for prediction
  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        body: formData,
      });
      const result: PredictionResult = await response.json();
      setPredictionResult(result);

      // Update viewer severity based on prediction
      if (result.prediction === 1 && result.affected_area) {
        setGradcamMean(result.confidence * (result.affected_area / 100));
      } else {
        setGradcamMean(0);
      }
    } catch (err) {
      console.error('Prediction failed:', err);
      alert('Prediction failed. Make sure the API server is running.');
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="app">
      <div className="panel">
        <h2 style={{ margin: 0 }}>AI Pneumonia Detection</h2>

        {/* Tab navigation */}
        <div style={{ display: 'flex', gap: 8, margin: '12px 0' }}>
          {(['viewer', 'upload', 'history'] as const).map(tab => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              style={{
                flex: 1,
                padding: '6px 8px',
                background: activeTab === tab ? '#4a9eff' : '#2a2a3a',
                color: '#fff',
                border: 'none',
                borderRadius: 6,
                cursor: 'pointer',
                fontSize: 12,
                fontWeight: activeTab === tab ? 700 : 400,
              }}
            >
              {tab === 'viewer' ? '🫁 3D View' : tab === 'upload' ? '📤 Upload' : '📋 History'}
            </button>
          ))}
        </div>

        {/* API Status indicator */}
        <div style={{ fontSize: 11, marginBottom: 8, display: 'flex', alignItems: 'center', gap: 6 }}>
          <span
            className="dot"
            style={{
              background: apiStatus === 'online' ? '#4CAF50' : apiStatus === 'offline' ? '#F44336' : '#FF9800',
              width: 8, height: 8, borderRadius: '50%', display: 'inline-block'
            }}
          />
          API: {apiStatus}
        </div>

        {activeTab === 'viewer' && (
          <>
            <div className="badge">
              <span className="dot" style={{ background: sevMeta.hex }} />
              <span>{sevMeta.label}</span>
              <span style={{ opacity: 0.75 }}>{severity.toFixed(2)}</span>
            </div>

            <label>Grad-CAM mean (0–1)</label>
            <input
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={gradcamMean}
              onChange={(e) => setGradcamMean(Number(e.target.value))}
            />

            <h3 style={{ color: '#ffffff', fontSize: '16px', margin: '20px 0 10px 0', fontWeight: '600' }}>Intensity Analysis</h3>
            <div className="kv">
              <div>Grad-CAM mean</div>
              <div style={{ color: '#4a9eff', fontWeight: '700', fontSize: '14px' }}>{gradcamMean.toFixed(3)}</div>
              <div>Severity (normalized)</div>
              <div style={{ color: sevMeta.hex, fontWeight: '700', fontSize: '14px' }}>{severity.toFixed(3)}</div>
              <div>Severity Level</div>
              <div style={{ color: sevMeta.hex, fontWeight: '700', fontSize: '14px' }}>{sevMeta.label}</div>
              <div>Model URL</div>
              <div style={{ color: '#ffffff', fontWeight: '600', fontSize: '11px' }}>lung_carcinoma.glb</div>
            </div>

            {/* Color Legend */}
            <h3 style={{ color: '#ffffff', fontSize: '14px', margin: '16px 0 8px 0', fontWeight: '600' }}>Color Legend</h3>
            <div style={{ fontSize: 12, lineHeight: 1.7, padding: '10px 12px', background: 'rgba(255,255,255,0.05)', borderRadius: 6, border: '1px solid rgba(255,255,255,0.1)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span style={{ width: 14, height: 14, borderRadius: 3, display: 'inline-block', background: sevMeta.hex, border: '1px solid rgba(255,255,255,0.2)', flexShrink: 0 }} />
                <span><strong style={{ color: sevMeta.hex }}>Infected area</strong> — highlighted region</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 4 }}>
                <span style={{ width: 14, height: 14, borderRadius: 3, display: 'inline-block', background: sevMeta.healthyHex, border: '1px solid rgba(255,255,255,0.2)', flexShrink: 0 }} />
                <span><strong style={{ color: sevMeta.healthyHex }}>Healthy tissue</strong> — tinted by severity</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 4 }}>
                <span style={{ width: 14, height: 14, borderRadius: 3, display: 'inline-block', background: 'linear-gradient(135deg, #ffb74d, #e53935)', border: '1px solid rgba(255,255,255,0.2)', flexShrink: 0 }} />
                <span><strong style={{ color: '#ffb74d' }}>Edge glow</strong> — infection boundary</span>
              </div>
            </div>

            <label>
              <input
                type="checkbox"
                checked={animate}
                onChange={(e) => setAnimate(e.target.checked)}
              />{' '}
              Animate progression
            </label>
          </>
        )}

        {activeTab === 'upload' && (
          <div style={{ marginTop: 12 }}>
            <h3 style={{ color: '#fff', fontSize: 14, marginBottom: 8 }}>Upload X-Ray for Analysis</h3>
            <input
              type="file"
              accept=".jpg,.jpeg,.png,.dcm"
              onChange={handleUpload}
              disabled={uploading || apiStatus !== 'online'}
              style={{ fontSize: 12, color: '#ccc' }}
            />
            {uploading && <p style={{ color: '#4a9eff', fontSize: 12 }}>Analyzing...</p>}

            {predictionResult && (
              <div style={{
                marginTop: 16,
                padding: 12,
                borderRadius: 8,
                background: predictionResult.prediction === 1 ? '#3a1a1a' : '#1a3a1a',
                border: `1px solid ${predictionResult.prediction === 1 ? '#F44336' : '#4CAF50'}`,
              }}>
                <h4 style={{ color: predictionResult.prediction === 1 ? '#F44336' : '#4CAF50', margin: '0 0 8px' }}>
                  {predictionResult.prediction_label}
                </h4>
                <div className="kv" style={{ fontSize: 12 }}>
                  <div>Confidence</div>
                  <div style={{ fontWeight: 700 }}>{(predictionResult.confidence * 100).toFixed(1)}%</div>
                  {predictionResult.severity && <>
                    <div>Severity</div>
                    <div style={{ fontWeight: 700 }}>{predictionResult.severity}</div>
                  </>}
                  {predictionResult.affected_area !== undefined && <>
                    <div>Affected Area</div>
                    <div style={{ fontWeight: 700 }}>{predictionResult.affected_area.toFixed(1)}%</div>
                  </>}
                </div>

                {predictionResult.recommendations && predictionResult.recommendations.length > 0 && (
                  <div style={{ marginTop: 8, fontSize: 11, color: '#ccc' }}>
                    <strong>Recommendations:</strong>
                    <ul style={{ margin: '4px 0', paddingLeft: 16 }}>
                      {predictionResult.recommendations.map((r, i) => (
                        <li key={i}>{r}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}

            {apiStatus !== 'online' && (
              <p style={{ color: '#F44336', fontSize: 11, marginTop: 8 }}>
                API is offline. Start it with: <code>python api.py</code>
              </p>
            )}
          </div>
        )}

        {activeTab === 'history' && (
          <div style={{ marginTop: 12 }}>
            <h3 style={{ color: '#fff', fontSize: 14, marginBottom: 8 }}>Recent Predictions</h3>
            <p style={{ color: '#999', fontSize: 11 }}>
              Patient history is available via the Streamlit dashboard or the REST API at <code>/patients</code>.
            </p>
          </div>
        )}

        <p style={{ fontSize: 12, opacity: 0.8, marginTop: 14, lineHeight: 1.35 }}>
          AI-powered pneumonia detection with Grad-CAM explainability, uncertainty estimation,
          and NLP report generation. For clinical use, always consult a radiologist.
        </p>
      </div>

      <div className="viewer">
        <ErrorBoundary>
          <React.Suspense fallback={null}>
            <LungViewer severity={severity} animate={animate} />
          </React.Suspense>
        </ErrorBoundary>
      </div>
    </div>
  );
}
