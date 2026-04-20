"""
Multilingual support module.
Provides translations for UI text, recommendations, and reports
in multiple languages.
"""

from typing import Dict, Optional


# Translation dictionaries for key UI and report strings
TRANSLATIONS = {
    'en': {
        # UI labels
        'app_title': 'AI-Powered Pneumonia Detection System',
        'upload_prompt': 'Upload Chest X-Ray Image',
        'prediction_results': 'Prediction Results',
        'pneumonia_detected': 'Pneumonia Detected',
        'normal_result': 'Normal',
        'confidence': 'Confidence',
        'severity_level': 'Severity Level',
        'affected_area': 'Affected Area',
        'gradcam_title': 'Grad-CAM Heatmap Analysis',
        'heatmap_overlay': 'Heatmap Overlay',
        'intensity_analysis': 'Intensity Analysis',
        'severity_assessment': 'Severity Assessment',
        '3d_visualization': '3D Lung Visualization',
        'recommendations_title': 'Medical Recommendations',
        'action_items': 'Action Items',
        'download_report': 'Download Full Report',
        'model_evaluation': 'Model Performance Evaluation',
        'about': 'About This System',
        'navigation': 'Navigation',
        'select_mode': 'Select Mode',
        'prediction': 'Prediction',
        'batch_processing': 'Batch Processing',
        'patient_history': 'Patient History',
        'comparison': 'Multi-Image Comparison',
        'settings': 'Settings',
        # Severity
        'normal': 'Normal',
        'mild': 'Mild',
        'moderate': 'Moderate', 
        'severe': 'Severe',
        # Urgency
        'low_urgency': 'Low Priority',
        'moderate_urgency': 'Moderate Priority',
        'high_urgency': 'High Priority',
        'critical_urgency': 'CRITICAL - Seek Immediate Attention',
        # Report
        'disclaimer': (
            'This AI analysis is for screening purposes only and should NOT replace '
            'professional medical diagnosis. Always consult qualified healthcare providers.'
        ),
        'report_title': 'CHEST X-RAY ANALYSIS REPORT',
        'findings': 'FINDINGS',
        'impression': 'IMPRESSION',
        'xai_method': 'Explanation Method',
        'uncertainty_warning': 'The model shows elevated uncertainty for this case.',
    },
    
    'es': {
        'app_title': 'Sistema de Detección de Neumonía con IA',
        'upload_prompt': 'Subir Imagen de Rayos X de Tórax',
        'prediction_results': 'Resultados de Predicción',
        'pneumonia_detected': 'Neumonía Detectada',
        'normal_result': 'Normal',
        'confidence': 'Confianza',
        'severity_level': 'Nivel de Gravedad',
        'affected_area': 'Área Afectada',
        'gradcam_title': 'Análisis de Mapa de Calor Grad-CAM',
        'heatmap_overlay': 'Superposición de Mapa de Calor',
        'intensity_analysis': 'Análisis de Intensidad',
        'severity_assessment': 'Evaluación de Gravedad',
        '3d_visualization': 'Visualización 3D de Pulmones',
        'recommendations_title': 'Recomendaciones Médicas',
        'action_items': 'Acciones a Tomar',
        'download_report': 'Descargar Informe Completo',
        'model_evaluation': 'Evaluación del Rendimiento del Modelo',
        'about': 'Acerca de Este Sistema',
        'navigation': 'Navegación',
        'select_mode': 'Seleccionar Modo',
        'prediction': 'Predicción',
        'batch_processing': 'Procesamiento por Lotes',
        'patient_history': 'Historial del Paciente',
        'comparison': 'Comparación Multi-Imagen',
        'settings': 'Configuración',
        'normal': 'Normal',
        'mild': 'Leve',
        'moderate': 'Moderado',
        'severe': 'Grave',
        'low_urgency': 'Prioridad Baja',
        'moderate_urgency': 'Prioridad Moderada',
        'high_urgency': 'Prioridad Alta',
        'critical_urgency': 'CRÍTICO - Busque Atención Inmediata',
        'disclaimer': (
            'Este análisis de IA es solo para fines de detección y NO debe reemplazar '
            'el diagnóstico médico profesional. Consulte siempre a profesionales de salud.'
        ),
        'report_title': 'INFORME DE ANÁLISIS DE RAYOS X DE TÓRAX',
        'findings': 'HALLAZGOS',
        'impression': 'IMPRESIÓN',
        'xai_method': 'Método de Explicación',
        'uncertainty_warning': 'El modelo muestra incertidumbre elevada para este caso.',
    },
    
    'fr': {
        'app_title': "Système de Détection de Pneumonie par IA",
        'upload_prompt': "Télécharger l'Image Radiographique Thoracique",
        'prediction_results': 'Résultats de Prédiction',
        'pneumonia_detected': 'Pneumonie Détectée',
        'normal_result': 'Normal',
        'confidence': 'Confiance',
        'severity_level': 'Niveau de Gravité',
        'affected_area': 'Zone Affectée',
        'gradcam_title': 'Analyse Grad-CAM',
        'heatmap_overlay': 'Superposition de Carte Thermique',
        'intensity_analysis': "Analyse d'Intensité",
        'severity_assessment': 'Évaluation de la Gravité',
        '3d_visualization': 'Visualisation 3D des Poumons',
        'recommendations_title': 'Recommandations Médicales',
        'action_items': 'Actions à Entreprendre',
        'download_report': 'Télécharger le Rapport Complet',
        'model_evaluation': 'Évaluation des Performances du Modèle',
        'about': 'À Propos',
        'navigation': 'Navigation',
        'select_mode': 'Sélectionner le Mode',
        'prediction': 'Prédiction',
        'batch_processing': 'Traitement par Lots',
        'patient_history': 'Historique du Patient',
        'comparison': 'Comparaison Multi-Images',
        'settings': 'Paramètres',
        'normal': 'Normal',
        'mild': 'Léger',
        'moderate': 'Modéré',
        'severe': 'Sévère',
        'low_urgency': 'Priorité Basse',
        'moderate_urgency': 'Priorité Modérée',
        'high_urgency': 'Priorité Élevée',
        'critical_urgency': "CRITIQUE - Consultez Immédiatement",
        'disclaimer': (
            "Cette analyse IA est uniquement destinée au dépistage et ne doit PAS remplacer "
            "un diagnostic médical professionnel."
        ),
        'report_title': 'RAPPORT D\'ANALYSE RADIOGRAPHIQUE THORACIQUE',
        'findings': 'RÉSULTATS',
        'impression': 'IMPRESSION',
        'xai_method': "Méthode d'Explication",
        'uncertainty_warning': 'Le modèle montre une incertitude élevée pour ce cas.',
    },
    
    'de': {
        'app_title': 'KI-gestütztes Pneumonie-Erkennungssystem',
        'upload_prompt': 'Röntgenbild des Brustkorbs hochladen',
        'prediction_results': 'Vorhersageergebnisse',
        'pneumonia_detected': 'Lungenentzündung erkannt',
        'normal_result': 'Normal',
        'confidence': 'Konfidenz',
        'severity_level': 'Schweregrad',
        'affected_area': 'Betroffene Fläche',
        'gradcam_title': 'Grad-CAM Heatmap-Analyse',
        'heatmap_overlay': 'Heatmap-Überlagerung',
        'intensity_analysis': 'Intensitätsanalyse',
        'severity_assessment': 'Schweregradbeurteilung',
        '3d_visualization': '3D-Lungenvisualisierung',
        'recommendations_title': 'Medizinische Empfehlungen',
        'action_items': 'Maßnahmen',
        'download_report': 'Vollständigen Bericht herunterladen',
        'model_evaluation': 'Modellleistungsbewertung',
        'about': 'Über dieses System',
        'navigation': 'Navigation',
        'select_mode': 'Modus wählen',
        'prediction': 'Vorhersage',
        'batch_processing': 'Stapelverarbeitung',
        'patient_history': 'Patientenhistorie',
        'comparison': 'Multi-Bild-Vergleich',
        'settings': 'Einstellungen',
        'normal': 'Normal',
        'mild': 'Leicht',
        'moderate': 'Mäßig',
        'severe': 'Schwer',
        'low_urgency': 'Niedrige Priorität',
        'moderate_urgency': 'Mittlere Priorität',
        'high_urgency': 'Hohe Priorität',
        'critical_urgency': 'KRITISCH - Sofortige Behandlung erforderlich',
        'disclaimer': (
            'Diese KI-Analyse dient nur zu Screening-Zwecken und sollte NICHT die '
            'professionelle medizinische Diagnose ersetzen.'
        ),
        'report_title': 'RÖNTGENANALYSE-BERICHT DES BRUSTKORBS',
        'findings': 'BEFUNDE',
        'impression': 'BEURTEILUNG',
        'xai_method': 'Erklärungsmethode',
        'uncertainty_warning': 'Das Modell zeigt erhöhte Unsicherheit für diesen Fall.',
    },
    
    'zh': {
        'app_title': 'AI驱动的肺炎检测系统',
        'upload_prompt': '上传胸部X光片',
        'prediction_results': '预测结果',
        'pneumonia_detected': '检测到肺炎',
        'normal_result': '正常',
        'confidence': '置信度',
        'severity_level': '严重程度',
        'affected_area': '受影响区域',
        'gradcam_title': 'Grad-CAM热力图分析',
        'heatmap_overlay': '热力图叠加',
        'intensity_analysis': '强度分析',
        'severity_assessment': '严重程度评估',
        '3d_visualization': '3D肺部可视化',
        'recommendations_title': '医疗建议',
        'action_items': '行动项目',
        'download_report': '下载完整报告',
        'model_evaluation': '模型性能评估',
        'about': '关于本系统',
        'navigation': '导航',
        'select_mode': '选择模式',
        'prediction': '预测',
        'batch_processing': '批量处理',
        'patient_history': '患者历史',
        'comparison': '多图像对比',
        'settings': '设置',
        'normal': '正常',
        'mild': '轻度',
        'moderate': '中度',
        'severe': '重度',
        'low_urgency': '低优先级',
        'moderate_urgency': '中等优先级',
        'high_urgency': '高优先级',
        'critical_urgency': '危急 - 请立即就医',
        'disclaimer': '本AI分析仅供筛查用途，不应替代专业医疗诊断。请始终咨询合格的医疗保健提供者。',
        'report_title': '胸部X光分析报告',
        'findings': '检查结果',
        'impression': '印象',
        'xai_method': '解释方法',
        'uncertainty_warning': '该模型对此病例显示较高的不确定性。',
    },
    
    'hi': {
        'app_title': 'AI-संचालित निमोनिया पहचान प्रणाली',
        'upload_prompt': 'छाती का एक्स-रे चित्र अपलोड करें',
        'prediction_results': 'भविष्यवाणी परिणाम',
        'pneumonia_detected': 'निमोनिया का पता चला',
        'normal_result': 'सामान्य',
        'confidence': 'विश्वास स्तर',
        'severity_level': 'गंभीरता स्तर',
        'affected_area': 'प्रभावित क्षेत्र',
        'gradcam_title': 'Grad-CAM हीटमैप विश्लेषण',
        'recommendations_title': 'चिकित्सा सिफारिशें',
        'action_items': 'कार्रवाई मद',
        'download_report': 'पूरी रिपोर्ट डाउनलोड करें',
        'normal': 'सामान्य',
        'mild': 'हल्का',
        'moderate': 'मध्यम',
        'severe': 'गंभीर',
        'disclaimer': 'यह AI विश्लेषण केवल स्क्रीनिंग उद्देश्यों के लिए है। कृपया योग्य डॉक्टर से परामर्श करें।',
    },

    'ar': {
        'app_title': 'نظام الكشف عن الالتهاب الرئوي بالذكاء الاصطناعي',
        'upload_prompt': 'تحميل صورة الأشعة السينية للصدر',
        'prediction_results': 'نتائج التنبؤ',
        'pneumonia_detected': 'تم اكتشاف الالتهاب الرئوي',
        'normal_result': 'طبيعي',
        'confidence': 'مستوى الثقة',
        'severity_level': 'مستوى الخطورة',
        'affected_area': 'المنطقة المتأثرة',
        'normal': 'طبيعي',
        'mild': 'خفيف',
        'moderate': 'متوسط',
        'severe': 'شديد',
        'disclaimer': 'هذا التحليل بالذكاء الاصطناعي هو لأغراض الفحص فقط. استشر طبيبك دائماً.',
    },

    'pt': {
        'app_title': 'Sistema de Detecção de Pneumonia com IA',
        'upload_prompt': 'Carregar Imagem de Raio-X do Tórax',
        'prediction_results': 'Resultados da Previsão',
        'pneumonia_detected': 'Pneumonia Detectada',
        'normal_result': 'Normal',
        'confidence': 'Confiança',
        'severity_level': 'Nível de Gravidade',
        'affected_area': 'Área Afetada',
        'normal': 'Normal',
        'mild': 'Leve',
        'moderate': 'Moderado',
        'severe': 'Grave',
        'disclaimer': 'Esta análise de IA é apenas para fins de triagem. Consulte sempre profissionais de saúde.',
    },

    'ja': {
        'app_title': 'AI肺炎検出システム',
        'upload_prompt': '胸部X線画像をアップロード',
        'prediction_results': '予測結果',
        'pneumonia_detected': '肺炎が検出されました',
        'normal_result': '正常',
        'confidence': '信頼度',
        'severity_level': '重症度',
        'affected_area': '影響範囲',
        'normal': '正常',
        'mild': '軽度',
        'moderate': '中等度',
        'severe': '重度',
        'disclaimer': 'このAI分析はスクリーニング目的のみです。必ず医療専門家に相談してください。',
    },

    'ko': {
        'app_title': 'AI 폐렴 감지 시스템',
        'upload_prompt': '흉부 X선 이미지 업로드',
        'prediction_results': '예측 결과',
        'pneumonia_detected': '폐렴 감지됨',
        'normal_result': '정상',
        'confidence': '신뢰도',
        'severity_level': '심각도',
        'affected_area': '영향 범위',
        'normal': '정상',
        'mild': '경미',
        'moderate': '중등도',
        'severe': '중증',
        'disclaimer': '이 AI 분석은 스크리닝 목적으로만 사용됩니다. 반드시 의료 전문가와 상담하세요.',
    },
}


def get_text(key: str, language: str = 'en', fallback: str = None) -> str:
    """
    Get translated text for a given key and language.
    Falls back to English if the key is not found in the target language.
    
    Args:
        key: Translation key
        language: Language code (e.g., 'en', 'es', 'fr')
        fallback: Fallback text if key not found anywhere
    
    Returns:
        Translated string
    """
    # Try target language
    lang_dict = TRANSLATIONS.get(language, {})
    if key in lang_dict:
        return lang_dict[key]
    
    # Fallback to English
    if language != 'en' and key in TRANSLATIONS.get('en', {}):
        return TRANSLATIONS['en'][key]
    
    # Fallback to key itself or provided fallback
    return fallback or key


def get_supported_languages() -> Dict[str, str]:
    """Return dict of language code → language name."""
    from config import MULTILINGUAL_CONFIG
    return MULTILINGUAL_CONFIG.get('language_names', {
        'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German',
        'zh': 'Chinese', 'hi': 'Hindi', 'ar': 'Arabic', 'pt': 'Portuguese',
        'ja': 'Japanese', 'ko': 'Korean'
    })


def translate_severity(severity: str, language: str = 'en') -> str:
    """Translate severity level."""
    key = severity.lower()
    return get_text(key, language, severity)


def translate_urgency(urgency: str, language: str = 'en') -> str:
    """Translate urgency level."""
    mapping = {
        'Low': 'low_urgency',
        'Moderate': 'moderate_urgency',
        'High': 'high_urgency',
        'Critical': 'critical_urgency',
    }
    key = mapping.get(urgency, urgency)
    return get_text(key, language, urgency)


if __name__ == "__main__":
    print("Multilingual support module loaded successfully")
    print(f"Supported languages: {list(TRANSLATIONS.keys())}")
    
    for lang in TRANSLATIONS:
        title = get_text('app_title', lang)
        print(f"  [{lang}] {title}")
