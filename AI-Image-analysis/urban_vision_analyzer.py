import os
import json
import logging
import tempfile
import httpx
from google.cloud import vision

# --------------------------------------------------
# GOOGLE CREDENTIALS — supports local file AND Vercel env var
# --------------------------------------------------
# Option 1 (local dev): JSON file next to this script
_CREDS_FILE = os.path.join(os.path.dirname(__file__), "instant-sound-456709-j3-9eba69829a2b.json")

# Option 2 (Vercel/production): base64-encoded JSON in env var GOOGLE_CREDENTIALS_JSON
_CREDS_JSON_ENV = os.environ.get("GOOGLE_CREDENTIALS_JSON", "")

if _CREDS_JSON_ENV:
    # Write credentials from env var to a temp file
    import base64
    try:
        _decoded = base64.b64decode(_CREDS_JSON_ENV).decode("utf-8")
        _tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        _tmp.write(_decoded)
        _tmp.close()
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = _tmp.name
        logging.info(f"[AI] Loaded credentials from GOOGLE_CREDENTIALS_JSON env var ✅")
    except Exception as e:
        logging.warning(f"[AI] Failed to load credentials from env var: {e}")
elif os.path.exists(_CREDS_FILE):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = _CREDS_FILE
    logging.info(f"[AI] Loaded credentials from file: {_CREDS_FILE} ✅")
else:
    logging.warning(f"[AI] No Google credentials found. Vision API will be unavailable.")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class UrbanIssueAnalyzer:
    """
    AI Image Analysis Engine using Google Cloud Vision.
    Accepts either a local file path OR a public URL.
    Returns issue_type, confidence, severity, department, and subcategory.
    """

    # --------------------------------------------------
    # AI ISSUE TYPE → APP DEPARTMENT + SUBCATEGORY MAP
    # --------------------------------------------------
    ISSUE_TO_DEPT_SUB = {
        "pothole":             ("Road & Infrastructure", "Pothole"),
        "garbage":             ("Waste Management",      "Garbage Heap"),
        "water_leakage":       ("Water Supply",          "Water Leakage"),
        "broken_pole":         ("Electricity",           "Exposed Wire"),
        "overflow_drain":      ("Sanitation",            "Clogged Sewer"),
        "streetlight_failure": ("Streetlight Maintenance", "Light Not Working"),
        "sanitation":          ("Sanitation",            "Public Toilet Issue"),
        "unknown":             (None, None),  # fallback → user selects manually
    }

    SUPPORTED_ISSUES = list(ISSUE_TO_DEPT_SUB.keys())

    LABEL_MAPPING = {
        "pothole":             ["pothole", "asphalt", "road", "crack", "rut", "damage", "pavement"],
        "garbage":             ["garbage", "waste", "trash", "litter", "dump", "debris", "rubbish"],
        "water_leakage":       ["leak", "leakage", "pipe", "puddle", "water spill", "flood"],
        "broken_pole":         ["electric pole", "utility pole", "power line", "wire", "electricity"],
        "overflow_drain":      ["drain", "sewer", "manhole", "overflow", "flooding", "stormwater"],
        "streetlight_failure": ["street light", "lamp post", "lighting", "streetlight"],
    }

    SANITATION_CONTEXT = [
        "toilet", "urinal", "washroom", "bathroom",
        "restroom", "latrine", "sanitary",
    ]

    SEVERITY_CONFIG = {
        "pothole":             (3, 8, 10),
        "garbage":             (2, 8, 12),
        "water_leakage":       (4, 8, 8),
        "broken_pole":         (8, 10, 5),
        "overflow_drain":      (8, 10, 5),
        "streetlight_failure": (5, 8, 6),
        "sanitation":          (6, 9, 6),
        "unknown":             (1, 3, 2),
    }

    def __init__(self):
        try:
            self.client = vision.ImageAnnotatorClient()
            logging.info("[AI] Google Vision client initialized ✅")
        except Exception as e:
            logging.warning(f"[AI] Vision client init failed: {e}")
            self.client = None

    # --------------------------------------------------
    def _detect_sanitation_context(self, labels):
        for label in labels:
            if any(kw in label.description.lower() for kw in self.SANITATION_CONTEXT):
                return True
        return False

    def _map_to_supported_issue(self, labels):
        scores = {issue: 0.0 for issue in self.SUPPORTED_ISSUES}
        for label in labels:
            desc = label.description.lower()
            conf = label.score
            for issue_type, keywords in self.LABEL_MAPPING.items():
                if any(kw in desc for kw in keywords):
                    scores[issue_type] = max(scores[issue_type], conf)
        best_issue, best_score = max(scores.items(), key=lambda x: x[1])
        return ("unknown", 0.0) if best_score == 0.0 else (best_issue, best_score)

    def _calculate_severity(self, issue_type, objects):
        base, max_sev, scale = self.SEVERITY_CONFIG.get(issue_type, (1, 3, 2))
        max_area = 0.0
        for obj in objects:
            verts = obj.bounding_poly.normalized_vertices
            if len(verts) == 4:
                w = max(v.x for v in verts) - min(v.x for v in verts)
                h = max(v.y for v in verts) - min(v.y for v in verts)
                max_area = max(max_area, w * h)
        if max_area == 0.0:
            max_area = 0.1
        return min(round(base + max_area * scale), max_sev)

    # --------------------------------------------------
    # ANALYZE FROM LOCAL FILE PATH
    # --------------------------------------------------
    def analyze_image(self, image_path: str) -> dict:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        with open(image_path, "rb") as f:
            content = f.read()
        return self._run_vision(content)

    # --------------------------------------------------
    # ANALYZE FROM PUBLIC URL (Supabase / any HTTP URL)
    # --------------------------------------------------
    def analyze_image_from_url(self, image_url: str) -> dict:
        logging.info(f"[AI] Downloading image from URL: {image_url}")
        try:
            response = httpx.get(image_url, timeout=20.0, follow_redirects=True)
            response.raise_for_status()
            content = response.content
            logging.info(f"[AI] Downloaded {len(content)} bytes ✅")
        except Exception as e:
            logging.error(f"[AI] Failed to download image: {e}")
            return self._unknown_result(str(e))
        return self._run_vision(content)

    # --------------------------------------------------
    # CORE VISION LOGIC
    # --------------------------------------------------
    def _run_vision(self, content: bytes) -> dict:
        if not self.client:
            logging.warning("[AI] Vision client not available, returning unknown")
            return self._unknown_result("Vision client not initialized")

        try:
            image = vision.Image(content=content)
            label_response = self.client.label_detection(image=image)
            object_response = self.client.object_localization(image=image)

            labels = label_response.label_annotations
            objects = object_response.localized_object_annotations

            logging.info(f"[AI] Labels detected: {[l.description for l in labels[:5]]}")

            sanitation_context = self._detect_sanitation_context(labels)
            issue_type, confidence = self._map_to_supported_issue(labels)
            confidence_percent = int(confidence * 100)

            if sanitation_context:
                issue_type = "sanitation"
                confidence_percent = max(confidence_percent, 80)

            if confidence_percent < 60:
                issue_type = "unknown"

            severity_score = self._calculate_severity(issue_type, objects)

            # Map to app department + subcategory
            department, subcategory = self.ISSUE_TO_DEPT_SUB.get(issue_type, (None, None))

            result = {
                "issue_type": issue_type,
                "confidence_percent": confidence_percent,
                "severity_score": severity_score,
                "department": department,
                "subcategory": subcategory,
                "ai_detected": issue_type != "unknown",
            }
            logging.info(f"[AI] Result: {result}")
            return result

        except Exception as e:
            logging.error(f"[AI] Vision analysis error: {e}")
            return self._unknown_result(str(e))

    def _unknown_result(self, error: str = "") -> dict:
        return {
            "issue_type": "unknown",
            "confidence_percent": 0,
            "severity_score": 1,
            "department": None,
            "subcategory": None,
            "ai_detected": False,
            "error": error,
        }


# --------------------------------------------------
# CLI TEST
# --------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="UrbanSathi Image Analyzer")
    parser.add_argument("--image", type=str, help="Local image path")
    parser.add_argument("--url", type=str, help="Public image URL")
    args = parser.parse_args()

    analyzer = UrbanIssueAnalyzer()
    if args.url:
        print(json.dumps(analyzer.analyze_image_from_url(args.url), indent=2))
    elif args.image:
        print(json.dumps(analyzer.analyze_image(args.image), indent=2))
    else:
        print("Provide --image or --url")