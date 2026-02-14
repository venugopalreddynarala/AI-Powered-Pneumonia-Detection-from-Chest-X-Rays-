import React, { useMemo, useState } from 'react';
import LungViewer from './LungViewer';
import ErrorBoundary from './ErrorBoundary';
import { normalizeGradcamStat, severityToColor } from './severity';

export default function App() {
  const [gradcamMean, setGradcamMean] = useState(0.35);
  const [animate, setAnimate] = useState(true);

  const severity = useMemo(() => {
    return normalizeGradcamStat(gradcamMean, 0.0, 1.0);
  }, [gradcamMean]);

  const sevMeta = severityToColor(severity);

  return (
    <div className="app">
      <div className="panel">
        <h2 style={{ margin: 0 }}>3D Lung Severity Viewer</h2>
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

        <label>
          <input
            type="checkbox"
            checked={animate}
            onChange={(e) => setAnimate(e.target.checked)}
          />{' '}
          Animate progression
        </label>

        <p style={{ fontSize: 12, opacity: 0.8, marginTop: 14, lineHeight: 1.35 }}>
          This visualization uses a static anatomical lung model with a severity-based surface intensity overlay for interpretability. Highlighted regions indicate relative severity, not anatomical localization.
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
