"""
Medical recommendations based on pneumonia severity and prediction results.
Provides actionable advice based on model predictions and Grad-CAM analysis.
"""

from typing import Dict, List, Tuple
import numpy as np


class SeverityClassifier:
    """Classify pneumonia severity from prediction confidence and affected area."""
    
    def __init__(self):
        # Severity thresholds
        self.confidence_thresholds = {
            'mild': (0.5, 0.7),
            'moderate': (0.7, 0.85),
            'severe': (0.85, 1.0)
        }
        
        self.area_thresholds = {
            'mild': (0, 20),      # 0-20% affected
            'moderate': (20, 40),  # 20-40% affected
            'severe': (40, 100)    # 40-100% affected
        }
    
    def classify(self, confidence: float, affected_area: float) -> str:
        """
        Classify severity based on confidence and affected area.
        
        Args:
            confidence: Model prediction confidence (0-1)
            affected_area: Percentage of affected lung tissue
            
        Returns:
            Severity level: 'Normal', 'Mild', 'Moderate', or 'Severe'
        """
        # If confidence is very low, likely normal
        if confidence < 0.5:
            return 'Normal'
        
        # Weight both confidence and affected area
        severity_score = (confidence * 0.6 + (affected_area / 100) * 0.4)
        
        if severity_score < 0.4:
            return 'Mild'
        elif severity_score < 0.65:
            return 'Moderate'
        else:
            return 'Severe'
    
    def get_affected_area_percentage(self, heatmap: np.ndarray, 
                                    threshold: float = 0.5) -> float:
        """
        Calculate percentage of affected lung area from heatmap.
        
        Args:
            heatmap: Grad-CAM heatmap (normalized 0-1)
            threshold: Intensity threshold for affected regions
            
        Returns:
            Percentage of affected area
        """
        affected_pixels = (heatmap > threshold).sum()
        total_pixels = heatmap.size
        percentage = (affected_pixels / total_pixels) * 100
        return percentage


def generate_recommendations(severity: str, 
                            confidence: float,
                            affected_area: float) -> Dict[str, any]:
    """
    Generate medical recommendations based on severity.
    
    Args:
        severity: Severity level (Normal/Mild/Moderate/Severe)
        confidence: Model confidence score
        affected_area: Percentage of affected lung area
        
    Returns:
        Dictionary with recommendations and urgency info
    """
    recommendations = {
        'severity': severity,
        'confidence': confidence,
        'affected_percentage': affected_area,
        'recommendations': [],
        'urgency_level': '',
        'follow_up': '',
        'warning': ''
    }
    
    if severity == 'Normal':
        recommendations['recommendations'] = [
            "No signs of pneumonia detected",
            "Maintain regular health checkups",
            "Practice good respiratory hygiene",
            "Stay up to date with vaccinations",
            "Monitor for any respiratory symptoms"
        ]
        recommendations['urgency_level'] = 'Low'
        recommendations['follow_up'] = 'Routine checkup in 6-12 months'
        recommendations['warning'] = 'If symptoms develop, consult a physician immediately'
        
    elif severity == 'Mild':
        recommendations['recommendations'] = [
            "Rest and stay hydrated",
            "Monitor symptoms closely (fever, cough, breathing difficulty)",
            "Consider consulting a physician for proper diagnosis",
            "Avoid strenuous activities",
            "Maintain good nutrition to support immune system",
            "Use over-the-counter fever reducers if needed (as per doctor's advice)",
            "Practice respiratory hygiene (cover coughs, wash hands)"
        ]
        recommendations['urgency_level'] = 'Moderate'
        recommendations['follow_up'] = 'Consult physician within 24-48 hours'
        recommendations['warning'] = 'Seek immediate care if symptoms worsen'
        
    elif severity == 'Moderate':
        recommendations['recommendations'] = [
            "⚠️ Consult a physician immediately for proper evaluation",
            "Likely requires prescription antibiotics or antiviral medication",
            "Rest and avoid physical exertion",
            "Stay well-hydrated (water, clear broths)",
            "Monitor vital signs (temperature, breathing rate, oxygen levels)",
            "May require chest X-ray confirmation and blood tests",
            "Follow prescribed treatment regimen strictly",
            "Consider hospitalization if symptoms progress"
        ]
        recommendations['urgency_level'] = 'High'
        recommendations['follow_up'] = 'Immediate medical consultation required (within 12-24 hours)'
        recommendations['warning'] = '🚨 Go to ER if experiencing severe breathing difficulty, chest pain, or high fever (>103°F)'
        
    else:  # Severe
        recommendations['recommendations'] = [
            "🚨 SEEK IMMEDIATE MEDICAL ATTENTION",
            "This may require hospitalization and intensive treatment",
            "Supplemental oxygen therapy may be necessary",
            "IV antibiotics and close monitoring required",
            "Chest X-ray, CT scan, and blood work needed",
            "Do NOT delay - go to emergency department or call ambulance",
            "Inform medical staff about this AI screening result",
            "May require ICU admission depending on respiratory status"
        ]
        recommendations['urgency_level'] = 'Critical'
        recommendations['follow_up'] = '🚑 EMERGENCY - Go to ER immediately or call 911'
        recommendations['warning'] = '⚠️ CRITICAL: This is a medical emergency. Severe pneumonia can be life-threatening without prompt treatment.'
    
    # Add general disclaimer
    recommendations['disclaimer'] = (
        "⚕️ IMPORTANT: This AI analysis is for screening purposes only and should NOT replace "
        "professional medical diagnosis. Always consult with qualified healthcare providers "
        "for accurate diagnosis and treatment planning."
    )
    
    return recommendations


def get_severity_color(severity: str) -> str:
    """
    Get color code for severity level.
    
    Args:
        severity: Severity level
        
    Returns:
        Hex color code
    """
    colors = {
        'Normal': '#00FF00',      # Green
        'Mild': '#FFFF00',        # Yellow
        'Moderate': '#FFA500',    # Orange
        'Severe': '#FF0000'       # Red
    }
    return colors.get(severity, '#808080')  # Gray as default


def get_urgency_icon(urgency_level: str) -> str:
    """
    Get emoji icon for urgency level.
    
    Args:
        urgency_level: Urgency level
        
    Returns:
        Emoji string
    """
    icons = {
        'Low': '✅',
        'Moderate': '⚠️',
        'High': '🚨',
        'Critical': '🚑'
    }
    return icons.get(urgency_level, '❓')


def generate_patient_report(severity: str,
                           confidence: float,
                           affected_area: float,
                           recommendations: Dict) -> str:
    """
    Generate a formatted patient report.
    
    Args:
        severity: Severity level
        confidence: Model confidence
        affected_area: Affected area percentage
        recommendations: Recommendations dictionary
        
    Returns:
        Formatted report as string
    """
    report = f"""
╔═══════════════════════════════════════════════════════════════╗
║          AI PNEUMONIA SCREENING REPORT                         ║
╚═══════════════════════════════════════════════════════════════╝

ANALYSIS RESULTS:
─────────────────────────────────────────────────────────────────
  Severity Level:        {severity}
  Model Confidence:      {confidence:.1%}
  Affected Lung Area:    {affected_area:.1f}%
  Urgency Level:         {recommendations['urgency_level']} {get_urgency_icon(recommendations['urgency_level'])}

RECOMMENDATIONS:
─────────────────────────────────────────────────────────────────
"""
    
    for i, rec in enumerate(recommendations['recommendations'], 1):
        report += f"  {i}. {rec}\n"
    
    report += f"""
FOLLOW-UP ACTION:
─────────────────────────────────────────────────────────────────
  {recommendations['follow_up']}

WARNING:
─────────────────────────────────────────────────────────────────
  {recommendations['warning']}

DISCLAIMER:
─────────────────────────────────────────────────────────────────
  {recommendations['disclaimer']}

═══════════════════════════════════════════════════════════════════
Generated by AI Pneumonia Detection System
═══════════════════════════════════════════════════════════════════
"""
    
    return report


def get_lifestyle_recommendations(severity: str) -> List[str]:
    """
    Get lifestyle and prevention recommendations.
    
    Args:
        severity: Severity level
        
    Returns:
        List of lifestyle recommendations
    """
    general = [
        "Get adequate sleep (7-9 hours per night)",
        "Maintain a balanced diet rich in vitamins",
        "Stay physically active (as health permits)",
        "Avoid smoking and secondhand smoke",
        "Practice good hand hygiene",
        "Stay up to date with vaccinations (flu, pneumonia vaccines)",
        "Manage chronic conditions (diabetes, COPD, etc.)"
    ]
    
    if severity in ['Moderate', 'Severe']:
        general.extend([
            "Complete full course of prescribed antibiotics",
            "Attend all follow-up appointments",
            "Use a humidifier to ease breathing",
            "Elevate head while sleeping",
            "Avoid alcohol and caffeine during recovery"
        ])
    
    return general


if __name__ == "__main__":
    # Test recommendations module
    print("Testing recommendations system...\n")
    
    # Test different severity levels
    test_cases = [
        ('Normal', 0.3, 5.0),
        ('Mild', 0.6, 15.0),
        ('Moderate', 0.78, 30.0),
        ('Severe', 0.95, 55.0)
    ]
    
    classifier = SeverityClassifier()
    
    for severity, conf, area in test_cases:
        recs = generate_recommendations(severity, conf, area)
        print(f"\n{'='*60}")
        print(f"Test Case: {severity}")
        print(f"{'='*60}")
        report = generate_patient_report(severity, conf, area, recs)
        print(report)
    
    print("\n✓ Recommendations module tested successfully!")
