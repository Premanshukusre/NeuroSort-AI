"""
NeuroSort AI: Refined Heuristic Detection Logic (V6.0).
Strictly follows user-provided keyword rules for high-confidence classification.
"""

def detect_document_type(text: str):
    """
    User-provided heuristic logic for strong keyword-based detection.
    Returns (Label, Score).
    """
    text = text.lower()

    # Internship Offer Letter
    if "offer letter" in text or "internship" in text:
        return "Internship Offer Letter", 0.9

    # Medical Report
    if "patient" in text or "diagnosis" in text or "blood pressure" in text:
        return "Medical Report", 0.9

    # Academic Assignment
    if "assignment" in text or "case study" in text or "practical" in text:
        return "Academic Assignment", 0.85

    # Financial Document
    if "invoice" in text or "bill" in text or "amount" in text:
        return "Financial Document", 0.85

    # Technical Documentation
    if "documentation" in text or "api" in text or "readme" in text or "developer" in text:
        return "Technical Documentation", 0.8

    # Default
    return None, 0
