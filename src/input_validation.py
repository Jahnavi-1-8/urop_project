"""Modular Input Validation Pipeline for Apple Leaf Images.

Multi-tier, generalised structural foliar verification:
1. Physical integrity & exposure: File decoding, dimensions, darkness, washout, and sharpness (Laplacian variance).
2. Boundary & contour analysis: Isolates candidate blade, measures outline continuity, contour dominance, and boundary roughness.
3. Scale-independent leaf geometry: Aspect ratio (length-to-width), bounding extent, solidity, compactness (isoperimetric quotient), and polygonal complexity (filtering rectangular documents, circular objects, and triangles).
4. Internal foliar structure: Lamina organic texture variance, curvilinear venation density (morphological top-hat/black-hat ridge filter), and central midrib longitudinal continuity.
5. Biological foliar corroboration: Multispectral foliar pigmentation (chlorophyll green, chlorotic yellow/rust, necrotic brown), human face/skin chrominance filtering, and synthetic grid/document discrimination.
6. Three explicit validation outcomes:
   - VALID: Sufficient structural evidence supports a leaf candidate (proceeds to pipeline).
   - INVALID: Evidence indicates non-leaf object or synthetic pattern (halts pipeline).
   - UNCERTAIN: Evidence is ambiguous, blurry, or occluded (halts pipeline).
"""

from typing import Dict, Any, Union, Tuple, Optional
import os
import sys
import io
import cv2
import numpy as np
from PIL import Image

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class ValidationStatus(str):
    """String subclass representing one of the three explicit validation outcomes:
    'VALID', 'INVALID', or 'UNCERTAIN', while maintaining full backward-compatibility
    with legacy status checks ('APPROVED', 'REJECTED', 'REVIEW_RECOMMENDED').
    """
    def __eq__(self, other: Any) -> bool:
        s = str(self)
        if s == "VALID" and other in ("VALID", "APPROVED"):
            return True
        if s == "INVALID" and other in ("INVALID", "REJECTED"):
            return True
        if s == "UNCERTAIN" and other in ("UNCERTAIN", "REVIEW_RECOMMENDED"):
            return True
        return super().__eq__(other)

    def __hash__(self) -> int:
        return hash(str(self))


class AppleLeafValidator:
    """Generalised structural validator for verifying apple leaf photographs.
    
    Evaluates multi-structural foliar evidence across independent characteristics:
    1. Leaf boundary continuity, contour coherence, and outline.
    2. Leaf shape & geometry (aspect ratio, extent, solidity, compactness, vertex complexity).
    3. Internal leaf structure (lamina organic texture variance, curvilinear venation, central midrib).
    4. Biological foliar corroboration (chlorophyll green, chlorotic rust, necrotic brown, skin/document filtering).
    5. Tri-state outcome: VALID (proceed), INVALID (halt), UNCERTAIN (halt).
    """

    def __init__(
        self,
        min_dim: int = 100,
        severe_blur_threshold: float = 16.0,
        moderate_blur_threshold: float = 28.0,
        min_candidate_ratio: float = 0.04,
        max_candidate_ratio: float = 0.98,
        face_min_skin_ratio: float = 0.35,
        face_max_foliar_ratio: float = 0.25,
        doc_max_foliar_ratio: float = 0.12,
        doc_min_neutral_ratio: float = 0.65,
    ):
        """Configurable thresholds documented for agronomic field deployment.

        Args:
            min_dim: Minimum acceptable image dimension in pixels (width or height).
            severe_blur_threshold: Hard rejection threshold for Laplacian sharpness variance.
            moderate_blur_threshold: Threshold below which an advisory review warning is recorded.
            min_candidate_ratio: Minimum area of primary candidate contour relative to image area.
            max_candidate_ratio: Maximum fraction of image area before full-frame checks trigger.
            face_min_skin_ratio: Minimum fraction of human skin chrominance required to confirm a face.
            face_max_foliar_ratio: Face candidate boxes exceeding this foliar ratio are treated as leaf spots.
            doc_max_foliar_ratio: Images exceeding this foliar ratio bypass document heuristics.
            doc_min_neutral_ratio: Minimum proportion of neutral paper background for document classification.
        """
        self.min_dim = min_dim
        self.severe_blur_threshold = severe_blur_threshold
        self.moderate_blur_threshold = moderate_blur_threshold
        self.min_candidate_ratio = min_candidate_ratio
        self.max_candidate_ratio = max_candidate_ratio
        self.face_min_skin_ratio = face_min_skin_ratio
        self.face_max_foliar_ratio = face_max_foliar_ratio
        self.doc_max_foliar_ratio = doc_max_foliar_ratio
        self.doc_min_neutral_ratio = doc_min_neutral_ratio

        # Load classical Haar Cascades for face detection
        cascade_dir = cv2.data.haarcascades
        self.frontal_face_cascade = cv2.CascadeClassifier(
            os.path.join(cascade_dir, "haarcascade_frontalface_default.xml")
        )
        self.profile_face_cascade = cv2.CascadeClassifier(
            os.path.join(cascade_dir, "haarcascade_profileface.xml")
        )

    def validate(
        self,
        image_input: Union[str, np.ndarray, Image.Image],
        is_rgb: bool = True,
    ) -> Dict[str, Any]:
        """Validate input image across structural, physical, and biological foliar cues.

        Returns structured dictionary:
            - is_valid: bool (True allows downstream classification/analysis, False halts pipeline)
            - status: ValidationStatus ("VALID" | "INVALID" | "UNCERTAIN")
            - reason: str (user-facing descriptive message)
            - details: Dict with fine-grained diagnostic telemetry
            - image_rgb: Validated RGB numpy array (if readable)
        """
        # --- CUE 1: File integrity and loading ---
        img_rgb, load_err = self._load_image(image_input, is_rgb=is_rgb)
        if img_rgb is None:
            return {
                "is_valid": False,
                "status": ValidationStatus("INVALID"),
                "reason": f"Corrupted or unreadable image file: {load_err}",
                "details": {"stage": "integrity", "error": load_err},
                "image_rgb": None,
            }

        h, w = img_rgb.shape[:2]
        if h < self.min_dim or w < self.min_dim:
            return {
                "is_valid": False,
                "status": ValidationStatus("INVALID"),
                "reason": f"Image resolution too low ({w}x{h}). Minimum required is {self.min_dim}x{self.min_dim}.",
                "details": {"stage": "dimensions", "width": w, "height": h},
                "image_rgb": img_rgb,
            }

        # --- CUE 2: Exposure and blur metrics ---
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        mean_intensity = float(np.mean(gray))
        std_intensity = float(np.std(gray))

        # Check for monochrome / grayscale representation
        is_monochrome = bool(
            np.array_equal(img_rgb[:, :, 0], img_rgb[:, :, 1])
            and np.array_equal(img_rgb[:, :, 1], img_rgb[:, :, 2])
        )

        # Severe darkness / pitch black
        if mean_intensity < 14.0:
            return {
                "is_valid": False,
                "status": ValidationStatus("INVALID"),
                "reason": "Image is severely underexposed (pitch black). Please provide an image with adequate lighting.",
                "details": {"stage": "exposure", "mean_intensity": mean_intensity},
                "image_rgb": img_rgb,
            }

        # Severe washout (pure white blank or zero dynamic contrast)
        if mean_intensity > 250.0 or std_intensity < 8.0:
            return {
                "is_valid": False,
                "status": ValidationStatus("INVALID"),
                "reason": "Image is washed out, blank, or lacks perceptible contrast.",
                "details": {"stage": "exposure", "mean_intensity": mean_intensity, "std_intensity": std_intensity},
                "image_rgb": img_rgb,
            }

        # Severe blur hard rejection
        if laplacian_var < self.severe_blur_threshold:
            return {
                "is_valid": False,
                "status": ValidationStatus("INVALID"),
                "reason": (
                    f"Image is heavily blurred or out of focus (sharpness score: {laplacian_var:.1f}, "
                    f"minimum threshold: {self.severe_blur_threshold:.1f}). Please capture a stabilized photo."
                ),
                "details": {"stage": "blur", "laplacian_var": round(laplacian_var, 2)},
                "image_rgb": img_rgb,
            }

        # --- CUE 3: Biological foliar pigmentation analysis ---
        foliar_mask, foliar_ratio, color_metrics = self._extract_foliar_pigmentation(img_rgb)

        # --- CUE 4: Cautious human face detection with skin verification ---
        face_confirmed, face_count, face_details = self._verify_human_face(img_rgb, gray, foliar_mask)
        if face_confirmed:
            return {
                "is_valid": False,
                "status": ValidationStatus("INVALID"),
                "reason": "Human face detected with confirmed facial skin characteristics. Non-leaf portraits are prohibited.",
                "details": {"stage": "human_face", "face_count": face_count, **face_details},
                "image_rgb": img_rgb,
            }

        # --- CUE 5: Document / Poster / Spreadsheet Check with orthogonal line awareness ---
        is_doc, doc_reason, doc_details = self._verify_document_or_synthetic(
            gray, img_rgb, foliar_mask, foliar_ratio, h, w
        )
        if is_doc:
            return {
                "is_valid": False,
                "status": ValidationStatus("INVALID"),
                "reason": f"Rejected as synthetic document or poster: {doc_reason}",
                "details": {"stage": "document_check", **doc_details},
                "image_rgb": img_rgb,
            }

        # Check for prominent non-foliar fruit (e.g. red apples on orchard branches)
        # Distinguish large 3D spherical fruit bodies from flat, irregular foliar disease lesions
        # (e.g. Cedar Apple Rust orange/red pustules or Black Rot frogeye lesions).
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        r1 = cv2.inRange(hsv, np.array([0, 75, 50]), np.array([10, 255, 255]))
        r2 = cv2.inRange(hsv, np.array([170, 75, 50]), np.array([180, 255, 255]))
        red_fruit_mask = cv2.bitwise_or(r1, r2)
        fruit_ratio = float(np.count_nonzero(red_fruit_mask) / (h * w))

        cnts_fruit, _ = cv2.findContours(red_fruit_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        has_large_round_fruit = False
        max_fruit_body_ratio = 0.0
        for fc in cnts_fruit:
            fa = float(cv2.contourArea(fc))
            fc_ratio = float(fa / (h * w))
            if fc_ratio > max_fruit_body_ratio:
                max_fruit_body_ratio = fc_ratio
            if fc_ratio >= 0.035:
                f_peri = cv2.arcLength(fc, True)
                f_circ = float((4.0 * np.pi * fa) / (f_peri**2 + 1e-6))
                if f_circ >= 0.20:
                    has_large_round_fruit = True

        is_fruit_scene = (fruit_ratio >= 0.11) or (fruit_ratio >= 0.08 and has_large_round_fruit)
        if is_fruit_scene:
            return {
                "is_valid": False,
                "status": ValidationStatus("INVALID"),
                "reason": (
                    f"Non-leaf orchard scene detected: Prominent fruit or non-foliar objects detected "
                    f"({fruit_ratio * 100:.1f}% fruit area). Please upload a focused photograph of a single leaf blade."
                ),
                "details": {
                    "stage": "fruit_detection",
                    "fruit_ratio": round(fruit_ratio, 4),
                    "max_fruit_body_ratio": round(max_fruit_body_ratio, 4),
                    "has_large_round_fruit": has_large_round_fruit,
                },
                "image_rgb": img_rgb,
            }

        # If RGB photograph lacks biological foliar colors across the entire image
        if not is_monochrome and foliar_ratio < 0.04:
            return {
                "is_valid": False,
                "status": ValidationStatus("INVALID"),
                "reason": (
                    f"Non-leaf object detected. Plant foliar pigmentation is negligible ({foliar_ratio * 100:.1f}% "
                    f"vs. expected minimum {self.min_candidate_ratio * 100:.1f}%). Image lacks natural leaf tissue."
                ),
                "details": {"stage": "foliar_pigmentation", "foliar_ratio": round(foliar_ratio, 4), **color_metrics},
                "image_rgb": img_rgb,
            }

        # --- CUE 6: Candidate Leaf Region Isolation ---
        candidate_mask, candidate_type = self._isolate_candidate_blade(
            img_rgb, gray, foliar_mask, is_monochrome
        )

        # --- CUE 7: Boundary and Contour Extraction ---
        contours, _ = cv2.findContours(candidate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {
                "is_valid": False,
                "status": ValidationStatus("INVALID"),
                "reason": "No coherent leaf contour or candidate blade could be detected.",
                "details": {"stage": "contour_detection", "foliar_ratio": round(foliar_ratio, 4)},
                "image_rgb": img_rgb,
            }

        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        primary_contour = contours[0]
        c_area = float(cv2.contourArea(primary_contour))
        total_pixels = float(h * w)
        c_ratio = float(c_area / total_pixels)

        if c_ratio < self.min_candidate_ratio:
            return {
                "is_valid": False,
                "status": ValidationStatus("UNCERTAIN") if c_ratio > 0.02 else ValidationStatus("INVALID"),
                "reason": (
                    f"Candidate foliar structure is too small or heavily occluded ({c_ratio * 100:.1f}% "
                    f"of frame vs. expected minimum {self.min_candidate_ratio * 100:.1f}%)."
                ),
                "details": {"stage": "candidate_size", "contour_area_ratio": round(c_ratio, 4)},
                "image_rgb": img_rgb,
            }

        # --- CUE 8: Leaf Shape & Geometry Analysis (Scale-Independent) ---
        geom = self._analyze_leaf_geometry(primary_contour, h, w, c_area, total_pixels)

        # --- CUE 9: Internal Leaf Structure (Lamina texture, venation, midrib) ---
        internal = self._analyze_internal_structure(gray, candidate_mask, primary_contour, geom["rect"])

        # --- CUE 10: Multi-Signal Decision Synthesis ---
        status, is_valid, reason, decision_details = self._synthesize_decision(
            geom=geom,
            internal=internal,
            gray=gray,
            laplacian_var=laplacian_var,
            foliar_ratio=foliar_ratio,
            color_metrics=color_metrics,
            is_monochrome=is_monochrome,
            h=h,
            w=w,
        )

        return {
            "is_valid": is_valid,
            "status": status,
            "reason": reason,
            "details": {
                "width": w,
                "height": h,
                "sharpness": round(laplacian_var, 2),
                "foliar_ratio": round(foliar_ratio, 4),
                **geom,
                **internal,
                **decision_details,
            },
            "image_rgb": img_rgb,
        }

    def _isolate_candidate_blade(
        self,
        img_rgb: np.ndarray,
        gray: np.ndarray,
        foliar_mask: np.ndarray,
        is_monochrome: bool,
    ) -> Tuple[np.ndarray, str]:
        """Isolate candidate leaf blade region using adaptive multi-representation cues."""
        h, w = gray.shape[:2]
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

        if not is_monochrome and np.count_nonzero(foliar_mask) > (0.03 * h * w):
            candidate = cv2.morphologyEx(foliar_mask, cv2.MORPH_CLOSE, kernel_close)
            candidate = self._fill_holes(candidate)
            return candidate, "color_foliar"

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        border_pixels = np.concatenate([otsu[0, :], otsu[-1, :], otsu[:, 0], otsu[:, -1]])
        if np.mean(border_pixels) > 127:
            candidate = cv2.bitwise_not(otsu)
        else:
            candidate = otsu

        candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel_close)
        candidate = self._fill_holes(candidate)
        return candidate, "intensity_contrast"

    def _analyze_leaf_geometry(
        self,
        contour: np.ndarray,
        h: int,
        w: int,
        c_area: float,
        total_pixels: float,
    ) -> Dict[str, Any]:
        """Evaluate scale-independent geometric properties of candidate leaf contour."""
        peri = cv2.arcLength(contour, True)
        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        hull_peri = float(cv2.arcLength(hull, True))

        rect = cv2.minAreaRect(contour)
        (cx, cy), (rw, rh), angle = rect
        major_axis = max(rw, rh)
        minor_axis = min(rw, rh)

        aspect_ratio = float(major_axis / (minor_axis + 1e-6))
        extent = float(c_area / (rw * rh + 1e-6))
        solidity = float(c_area / (hull_area + 1e-6))
        compactness = float((4.0 * np.pi * c_area) / (peri ** 2 + 1e-6))
        boundary_roughness = float(peri / (hull_peri + 1e-6))

        approx_poly = cv2.approxPolyDP(contour, 0.02 * peri, True)
        vtx_count = len(approx_poly)

        is_rectangle = False
        is_full_frame = (c_area / total_pixels) >= 0.80
        if vtx_count == 4 and extent > 0.88 and solidity > 0.94 and not is_full_frame:
            is_rectangle = True

        is_circle = False
        if compactness >= 0.84 and aspect_ratio <= 1.08:
            is_circle = True

        is_triangle = (vtx_count == 3)

        # Check convexity defects (deep concavities between branches/overlapping leaves)
        hull_idx = cv2.convexHull(contour, returnPoints=False)
        max_defect_ratio = 0.0
        if len(contour) > 3 and len(hull_idx) > 3:
            defects = cv2.convexityDefects(contour, hull_idx)
            if defects is not None:
                depths = defects[:, 0, 3] / 256.0
                max_defect_ratio = float(np.max(depths) / (minor_axis + 1e-6))

        return {
            "contour_area_ratio": round(c_area / total_pixels, 4),
            "aspect_ratio": round(aspect_ratio, 3),
            "extent": round(extent, 3),
            "solidity": round(solidity, 3),
            "compactness": round(compactness, 3),
            "boundary_roughness": round(boundary_roughness, 3),
            "vertex_count": vtx_count,
            "max_defect_ratio": round(max_defect_ratio, 3),
            "is_rectangle": is_rectangle,
            "is_circle": is_circle,
            "is_triangle": is_triangle,
            "rect": rect,
        }

    def _analyze_internal_structure(
        self,
        gray: np.ndarray,
        candidate_mask: np.ndarray,
        contour: np.ndarray,
        rect: Tuple[Tuple[float, float], Tuple[float, float], float],
    ) -> Dict[str, Any]:
        """Examine lamina organic texture, curvilinear venation, and central midrib."""
        h, w = gray.shape[:2]

        blade_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(blade_mask, [contour], -1, 255, -1)

        # Compute black void ratio and foliar tissue fill within the candidate blade
        blade_px_all = gray[blade_mask > 0]
        black_in_blade = float(np.count_nonzero(blade_px_all < 15) / (len(blade_px_all) + 1e-6))
        foliar_in_blade = (
            float(np.count_nonzero(candidate_mask[blade_mask > 0]) / (len(blade_px_all) + 1e-6))
        )

        eroded_mask = cv2.erode(blade_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        sample_mask = eroded_mask if np.count_nonzero(eroded_mask) > 100 else blade_mask
        blade_pixels = gray[sample_mask > 0]

        lamina_std = float(np.std(blade_pixels)) if len(blade_pixels) > 0 else 0.0
        lamina_mean = float(np.mean(blade_pixels)) if len(blade_pixels) > 0 else 0.0

        top = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
        black = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
        ridge = cv2.max(top, black)

        ridge_blade = ridge[sample_mask > 0]
        vein_mean_strength = float(np.mean(ridge_blade)) if len(ridge_blade) > 0 else 0.0
        vein_pixel_pct = (
            float(np.count_nonzero(ridge_blade > 10) / (len(ridge_blade) + 1e-6)) * 100.0
        )

        (cx, cy), (rw, rh), angle = rect
        if rw < rh:
            angle = angle + 90
            rw, rh = rh, rw

        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        rot_ridge = cv2.warpAffine(ridge, M, (w, h))
        rot_mask = cv2.warpAffine(blade_mask, M, (w, h))

        y1 = max(0, int(cy - 0.12 * rh))
        y2 = min(h, int(cy + 0.12 * rh))
        x1 = max(0, int(cx - 0.40 * rw))
        x2 = min(w, int(cx + 0.40 * rw))

        midrib_continuity = 0.0
        midrib_strength = 0.0
        if y2 > y1 and x2 > x1:
            strip_ridge = rot_ridge[y1:y2, x1:x2]
            strip_mask = rot_mask[y1:y2, x1:x2]
            valid_px = strip_ridge[strip_mask > 0]
            if len(valid_px) > 0:
                midrib_strength = float(np.mean(valid_px))
                col_max = np.max(strip_ridge, axis=0)
                col_mask = np.max(strip_mask, axis=0) > 0
                valid_cols = col_max[col_mask]
                if len(valid_cols) > 0:
                    midrib_continuity = float(np.count_nonzero(valid_cols > 8) / len(valid_cols))

        return {
            "lamina_std": round(lamina_std, 2),
            "lamina_mean": round(lamina_mean, 2),
            "black_in_blade": round(black_in_blade, 3),
            "foliar_in_blade": round(foliar_in_blade, 3),
            "vein_mean_strength": round(vein_mean_strength, 2),
            "vein_pixel_pct": round(vein_pixel_pct, 2),
            "midrib_continuity": round(midrib_continuity, 3),
            "midrib_strength": round(midrib_strength, 2),
        }

    def _synthesize_decision(
        self,
        geom: Dict[str, Any],
        internal: Dict[str, Any],
        gray: np.ndarray,
        laplacian_var: float,
        foliar_ratio: float,
        color_metrics: Dict[str, Any],
        is_monochrome: bool,
        h: int,
        w: int,
    ) -> Tuple[ValidationStatus, bool, str, Dict[str, Any]]:
        """Combine geometric, textural, venation, and foliar evidence into a robust final outcome."""
        unique_levels = len(np.unique(gray))
        is_binary_or_wire = (unique_levels <= 4)

        # 1. HARD REJECTIONS: Clear geometric or synthetic non-leaves
        if is_binary_or_wire:
            return (
                ValidationStatus("INVALID"),
                False,
                f"Non-leaf representation detected: Image is a thresholded binary mask, edge map, or skeleton ({unique_levels} intensity levels) lacking continuous foliar lamina tissue.",
                {"rejection_rule": "binary_or_wire_representation", "unique_levels": unique_levels},
            )

        # Check whether candidate possesses genuine organic foliar lamina texture and venation
        has_organic_blade_tissue = (
            internal.get("lamina_std", 0.0) >= 10.0
            and (internal.get("vein_pixel_pct", 0.0) >= 7.0 or internal.get("lamina_std", 0.0) >= 20.0)
            and foliar_ratio >= 0.35
            and internal.get("foliar_in_blade", 1.0) >= 0.70
        )

        if geom["is_rectangle"] and not has_organic_blade_tissue:
            return (
                ValidationStatus("INVALID"),
                False,
                "Non-leaf object detected: Rectangular artificial outline, card, or document boundary.",
                {"rejection_rule": "geometric_rectangle"},
            )

        if geom["is_circle"]:
            return (
                ValidationStatus("INVALID"),
                False,
                "Non-leaf object detected: Circular synthetic shape or non-blade spherical object.",
                {"rejection_rule": "geometric_circle"},
            )

        if geom["is_triangle"]:
            return (
                ValidationStatus("INVALID"),
                False,
                "Non-leaf object detected: Triangular artificial polygon.",
                {"rejection_rule": "geometric_triangle"},
            )

        # Extreme extent: solid box/card (unless confirmed biological leaf filling frame)
        if geom["extent"] > 0.92 and not has_organic_blade_tissue:
            return (
                ValidationStatus("INVALID"),
                False,
                f"Non-leaf object detected: Overly rectangular candidate fill (extent: {geom['extent']:.2f}).",
                {"rejection_rule": "excessive_extent"},
            )

        # Flat synthetic fill with zero internal texture
        if internal["lamina_std"] < 3.5 and internal["vein_pixel_pct"] < 3.0:
            return (
                ValidationStatus("INVALID"),
                False,
                f"Non-leaf object detected: Flat synthetic fill lacking natural biological lamina texture (std: {internal['lamina_std']:.1f}).",
                {"rejection_rule": "flat_synthetic_fill"},
            )

        # Extreme linear wire / border / margin
        if geom["aspect_ratio"] > 5.5:
            return (
                ValidationStatus("INVALID"),
                False,
                f"Non-leaf object detected: Candidate aspect ratio ({geom['aspect_ratio']:.1f}) is excessively elongated for an apple leaf.",
                {"rejection_rule": "extreme_aspect_ratio"},
            )

        # Severe fragmentation / low solidity
        if geom["solidity"] < 0.45:
            return (
                ValidationStatus("INVALID"),
                False,
                f"Non-leaf object detected: Foliar candidate is overly fragmented or disconnected (solidity: {geom['solidity']:.2f}).",
                {"rejection_rule": "low_solidity"},
            )

        # Rejection for hollow / perforated digital matrix code, scattered glowing dots, or excessive black void
        if not is_monochrome:
            black_in_blade = internal.get("black_in_blade", 0.0)
            foliar_in_blade = internal.get("foliar_in_blade", 1.0)
            lamina_mean = internal.get("lamina_mean", 100.0)
            if black_in_blade > 0.18 or foliar_in_blade < 0.72 or lamina_mean < 35.0:
                return (
                    ValidationStatus("INVALID"),
                    False,
                    (
                        f"Non-leaf digital pattern detected: Perforated/hollow candidate structure "
                        f"({black_in_blade * 100:.1f}% black void, {foliar_in_blade * 100:.1f}% foliar fill, mean intensity {lamina_mean:.1f}). "
                        "Candidate lacks continuous biological leaf lamina tissue."
                    ),
                    {"rejection_rule": "digital_void_or_matrix_pattern", **internal},
                )

        # Check for wide-angle tree canopy / branches with deep concavity defects
        max_defect_ratio = geom.get("max_defect_ratio", 0.0)
        if max_defect_ratio > 0.78:
            return (
                ValidationStatus("INVALID"),
                False,
                (
                    f"Non-leaf orchard canopy detected: Deeply concaved branch or tree canopy structure "
                    f"(defect ratio: {max_defect_ratio:.2f}). Please upload a focused photograph of a single leaf blade."
                ),
                {"rejection_rule": "canopy_clutter", "max_defect_ratio": max_defect_ratio},
            )

        # 2. SCORING STRUCTURAL PILLARS
        geom_score = 0.0
        if 1.05 <= geom["aspect_ratio"] <= 3.8:
            geom_score += 0.25
        if 0.40 <= geom["extent"] <= 0.88:
            geom_score += 0.25
        if 0.65 <= geom["solidity"] <= 0.98:
            geom_score += 0.25
        if 0.12 <= geom["compactness"] <= 0.78:
            geom_score += 0.25

        internal_score = 0.0
        if internal["lamina_std"] >= 12.0:
            internal_score += 0.35
        elif internal["lamina_std"] >= 6.0:
            internal_score += 0.20

        if internal["vein_pixel_pct"] >= 15.0:
            internal_score += 0.35
        elif internal["vein_pixel_pct"] >= 6.0:
            internal_score += 0.20

        if internal["midrib_continuity"] >= 0.30:
            internal_score += 0.30
        elif internal["midrib_continuity"] >= 0.12:
            internal_score += 0.15

        bio_score = 0.0
        if not is_monochrome:
            if foliar_ratio >= 0.10:
                bio_score = 1.0
            elif foliar_ratio >= 0.05:
                bio_score = 0.7
            else:
                bio_score = 0.3
        else:
            bio_score = 0.5

        structural_confidence = (
            0.40 * geom_score + 0.40 * internal_score + 0.20 * bio_score
        )

        # 3. UNCERTAINTY CONDITIONS
        is_moderate_blur = (laplacian_var < self.moderate_blur_threshold)
        is_sparse_foliar = (foliar_ratio < 0.08 and not is_monochrome)
        is_ambiguous_structure = (0.40 <= structural_confidence < 0.62)

        if is_monochrome:
            if structural_confidence >= 0.70:
                return (
                    ValidationStatus("VALID"),
                    True,
                    "Input verified as a structurally plausible leaf blade photograph (grayscale/monochrome). Note: Physical foliar pigmentation cannot be confirmed without color.",
                    {"structural_confidence": round(structural_confidence, 3), "representation": "monochrome"},
                )
            else:
                return (
                    ValidationStatus("UNCERTAIN"),
                    False,
                    "Monochrome/grayscale input lacks conclusive structural evidence of a genuine leaf blade and lacks biological foliar color confirmation.",
                    {"structural_confidence": round(structural_confidence, 3), "representation": "monochrome"},
                )

        if is_moderate_blur or is_sparse_foliar or is_ambiguous_structure:
            if structural_confidence < 0.50:
                return (
                    ValidationStatus("INVALID"),
                    False,
                    "Non-leaf object detected: Candidate lacks coherent leaf geometry and internal venation structure.",
                    {"structural_confidence": round(structural_confidence, 3)},
                )
            else:
                return (
                    ValidationStatus("UNCERTAIN"),
                    False,
                    (
                        "Input structure is ambiguous or borderline. Blurriness, partial occlusion, or complex lighting "
                        f"(structural confidence: {structural_confidence:.2f}, sharpness: {laplacian_var:.1f}). "
                        "Please capture a clearer, centered photograph of the leaf."
                    ),
                    {"structural_confidence": round(structural_confidence, 3), "is_blur": is_moderate_blur},
                )

        # 4. FINAL APPROVAL (VALID)
        if structural_confidence >= 0.62:
            return (
                ValidationStatus("VALID"),
                True,
                "Input verified as a valid, clear apple leaf photograph.",
                {"structural_confidence": round(structural_confidence, 3)},
            )
        else:
            return (
                ValidationStatus("INVALID"),
                False,
                f"Non-leaf object detected: Candidate lacks structural consistency with an apple leaf blade (confidence: {structural_confidence:.2f}).",
                {"structural_confidence": round(structural_confidence, 3)},
            )

    def _fill_holes(self, binary_mask: np.ndarray) -> np.ndarray:
        """Fill internal holes within a binary mask using border-padded flood-fill."""
        if binary_mask is None or binary_mask.size == 0 or np.count_nonzero(binary_mask) == 0:
            return binary_mask
        h, w = binary_mask.shape[:2]
        padded = cv2.copyMakeBorder(binary_mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        floodfilled = padded.copy()
        mask_ff = np.zeros((h + 4, w + 4), np.uint8)
        cv2.floodFill(floodfilled, mask_ff, (0, 0), 255)
        floodfilled_inv = cv2.bitwise_not(floodfilled)
        filled = padded | floodfilled_inv
        return filled[1:h + 1, 1:w + 1]

    def _load_image(
        self,
        image_input: Union[str, np.ndarray, Image.Image],
        is_rgb: bool,
    ) -> Tuple[Optional[np.ndarray], Optional[str]]:
        try:
            if isinstance(image_input, str):
                if not os.path.exists(image_input):
                    return None, f"File does not exist: {image_input}"
                pil_img = Image.open(image_input)
                pil_img.verify()
                pil_img = Image.open(image_input).convert("RGB")
                return np.array(pil_img, dtype=np.uint8), None

            elif isinstance(image_input, Image.Image):
                rgb_img = image_input.convert("RGB")
                return np.array(rgb_img, dtype=np.uint8), None

            elif isinstance(image_input, np.ndarray):
                if image_input.size == 0 or image_input.ndim not in [2, 3]:
                    return None, "Empty or invalid numpy array shape"
                if image_input.ndim == 2:
                    return cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB), None
                if is_rgb:
                    return image_input.astype(np.uint8), None
                return cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB), None
            else:
                return None, f"Unsupported input type: {type(image_input)}"
        except Exception as e:
            return None, str(e)

    def _extract_foliar_pigmentation(
        self, img_rgb: np.ndarray
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """Segment natural plant foliar pigments: green chlorophyll, chlorotic rust/yellow, and necrotic lesions."""
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        total_px = float(img_rgb.shape[0] * img_rgb.shape[1])

        # 1. Healthy green chlorophyll (Hue 20 to 100, moderate saturation & value)
        green_mask = cv2.inRange(hsv, np.array([20, 18, 18]), np.array([100, 255, 255]))

        # 2. Chlorotic rust / bright yellow (Hue 8 to 28, sat >= 22, val >= 22)
        yellow_rust_mask = cv2.inRange(hsv, np.array([8, 22, 22]), np.array([28, 255, 255]))

        # 3. Necrotic lesions / dark spots within biological range (Hue 4-28, moderate sat)
        brown_necrotic_mask = cv2.inRange(hsv, np.array([4, 12, 12]), np.array([28, 255, 140]))

        # Combined foliar mask
        foliar_mask = cv2.bitwise_or(green_mask, yellow_rust_mask)
        foliar_mask = cv2.bitwise_or(foliar_mask, brown_necrotic_mask)

        foliar_count = int(np.count_nonzero(foliar_mask))
        foliar_ratio = float(foliar_count / (total_px + 1e-6))

        return foliar_mask, foliar_ratio, {
            "green_ratio": float(np.count_nonzero(green_mask) / total_px),
            "yellow_rust_ratio": float(np.count_nonzero(yellow_rust_mask) / total_px),
            "brown_necrotic_ratio": float(np.count_nonzero(brown_necrotic_mask) / total_px),
        }

    def _verify_human_face(
        self,
        img_rgb: np.ndarray,
        gray: np.ndarray,
        foliar_mask: np.ndarray,
    ) -> Tuple[bool, int, Dict[str, Any]]:
        """Cautious face detection: filters out false Haar responses on leaf lesions and veins."""
        faces = self.frontal_face_cascade.detectMultiScale(
            gray, scaleFactor=1.15, minNeighbors=7, minSize=(60, 60)
        )

        profile_faces = self.profile_face_cascade.detectMultiScale(
            gray, scaleFactor=1.15, minNeighbors=7, minSize=(60, 60)
        )

        all_candidates = list(faces) if len(faces) > 0 else []
        if len(profile_faces) > 0:
            all_candidates.extend(list(profile_faces))

        if not all_candidates:
            return False, 0, {}

        img_ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
        confirmed_faces = 0
        details = []

        for (fx, fy, fw, fh) in all_candidates:
            face_roi_foliar = foliar_mask[fy : fy + fh, fx : fx + fw]
            face_roi_ycrcb = img_ycrcb[fy : fy + fh, fx : fx + fw]
            face_area = float(fw * fh)

            foliar_in_face = float(np.count_nonzero(face_roi_foliar) / (face_area + 1e-6))
            if foliar_in_face > self.face_max_foliar_ratio:
                continue

            cr = face_roi_ycrcb[:, :, 1]
            cb = face_roi_ycrcb[:, :, 2]
            skin_mask = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)
            skin_ratio = float(np.count_nonzero(skin_mask) / (face_area + 1e-6))

            if skin_ratio >= self.face_min_skin_ratio:
                confirmed_faces += 1
                details.append({
                    "box": (int(fx), int(fy), int(fw), int(fh)),
                    "skin_ratio": round(skin_ratio, 3),
                    "foliar_ratio": round(foliar_in_face, 3),
                })

        if confirmed_faces > 0:
            return True, confirmed_faces, {"confirmed_faces": details}

        return False, 0, {"filtered_spurious_candidates": len(all_candidates)}

    def _verify_document_or_synthetic(
        self,
        gray: np.ndarray,
        img_rgb: np.ndarray,
        foliar_mask: np.ndarray,
        foliar_ratio: float,
        h: int,
        w: int,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Multi-cue document / poster / spreadsheet detection with leaf vein awareness."""
        total_pixels = float(h * w)

        if foliar_ratio >= self.doc_max_foliar_ratio:
            return (
                False,
                "",
                {"status": "bypassed_due_to_confirmed_foliar_tissue", "foliar_ratio": foliar_ratio},
            )

        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        neutral_paper_mask = (sat <= 35) & (val >= 150)
        neutral_paper_ratio = float(np.count_nonzero(neutral_paper_mask) / total_pixels)

        edges = cv2.Canny(gray, 60, 160)
        min_line_len = int(min(h, w) * 0.40)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=100, minLineLength=min_line_len, maxLineGap=12
        )

        orthogonal_lines = 0
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                dx = abs(x2 - x1)
                dy = abs(y2 - y1)
                if dx < 2 or dy < 2:
                    orthogonal_lines += 1

        if neutral_paper_ratio >= self.doc_min_neutral_ratio and foliar_ratio < 0.06:
            if orthogonal_lines >= 8:
                return (
                    True,
                    "Rectilinear document table grid or paper border detected on neutral background.",
                    {
                        "neutral_paper_ratio": round(neutral_paper_ratio, 3),
                        "foliar_ratio": round(foliar_ratio, 3),
                        "orthogonal_lines": orthogonal_lines,
                    },
                )

        edge_density = float(np.count_nonzero(edges) / total_pixels)
        if neutral_paper_ratio > 0.82 and foliar_ratio < 0.04 and edge_density > 0.045:
            return (
                True,
                "Dense text paragraphs on white paper background detected.",
                {
                    "neutral_paper_ratio": round(neutral_paper_ratio, 3),
                    "edge_density": round(edge_density, 3),
                    "foliar_ratio": round(foliar_ratio, 3),
                },
            )

        return False, "", {
            "neutral_paper_ratio": round(neutral_paper_ratio, 3),
            "orthogonal_lines": orthogonal_lines,
            "foliar_ratio": round(foliar_ratio, 3),
        }

    def _verify_blade_structure(
        self, foliar_mask: np.ndarray, h: int, w: int
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Maintained helper method for backward compatibility."""
        total_pixels = float(h * w)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        foliar_closed = cv2.morphologyEx(foliar_mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(foliar_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return False, "No cohesive foliar contour detected.", {}

        largest_contour = max(contours, key=cv2.contourArea)
        c_area = float(cv2.contourArea(largest_contour))
        c_ratio = float(c_area / total_pixels)

        if c_ratio < self.min_candidate_ratio:
            return (
                False,
                f"Foliar structure too small ({c_ratio * 100:.1f}% vs. required {self.min_candidate_ratio * 100:.1f}%).",
                {"contour_area_ratio": round(c_ratio, 4)},
            )

        hull = cv2.convexHull(largest_contour)
        hull_area = float(cv2.contourArea(hull))
        solidity = float(c_area / (hull_area + 1e-6))

        if solidity < 0.45:
            return (
                False,
                f"Foliar object is overly fragmented or non-cohesive (solidity: {solidity:.2f}).",
                {"solidity": round(solidity, 3), "contour_area_ratio": round(c_ratio, 4)},
            )

        return (
            True,
            "Valid organic leaf blade structure.",
            {"contour_area_ratio": round(c_ratio, 4), "solidity": round(solidity, 3)},
        )


# Global singleton validator
default_validator = AppleLeafValidator()


def validate_apple_leaf_input(
    image_input: Union[str, np.ndarray, Image.Image],
    is_rgb: bool = True,
) -> Dict[str, Any]:
    """Top-level convenience validator function."""
    return default_validator.validate(image_input, is_rgb=is_rgb)
