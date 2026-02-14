export function clamp01(v: number): number {
  return Math.max(0, Math.min(1, v));
}

export function normalizeGradcamStat(stat: number, inMin = 0.0, inMax = 1.0): number {
  if (!Number.isFinite(stat)) return 0;
  if (inMax <= inMin) return 0;
  return clamp01((stat - inMin) / (inMax - inMin));
}

export function severityToColor(sev: number): { hex: string; label: string } {
  if (sev >= 0.7) return { hex: '#e53935', label: 'Severe' };
  if (sev >= 0.4) return { hex: '#ff9800', label: 'Moderate' };
  if (sev >= 0.2) return { hex: '#ffd54f', label: 'Mild' };
  return { hex: '#90caf9', label: 'Low' };
}
