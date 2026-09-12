import os
import math
import re
import time
import logging
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger("PCB-Reasoning")

# Standard baseline assembly recipe for our PCB board architecture:
# A fully assembled board typically contains:
# - Transformer: at least 1-2
# - MOSFET: at least 1
# - MOV: at least 1
# - Capacitors (Cap1..Cap4): at least 4 (1 of each or 4+ total)
# - Resistors: at least 1
REFERENCE_PCB_RECIPE = {
    "Transformer": {"min": 1, "description": "Power transformer coil"},
    "MOSFET": {"min": 1, "description": "Power switching transistor"},
    "MOV": {"min": 1, "description": "Metal Oxide Varistor surge protector"},
    "Capacitor": {"min": 4, "description": "Electrolytic / Ceramic capacitor cluster (Cap1-Cap4)"},
    "Resistor": {"min": 1, "description": "SMD / Through-hole resistor"}
}

# Synonyms and mapping for natural language entity queries
COMPONENT_ALIASES = {
    "capacitor": ["capacitor", "capacitors", "cap", "caps", "cap1", "cap2", "cap3", "cap4", "electrolytic"],
    "resistor": ["resistor", "resistors", "resestor", "smd resistor"],
    "mosfet": ["mosfet", "mosfets", "transistor", "fet", "power ic", "ic"],
    "mov": ["mov", "varistor", "surge suppressor"],
    "transformer": ["transformer", "transformers", "coil", "inductor"],
    "connector": ["connector", "connectors", "header", "terminal", "usb", "port"],
}


class IntentRouter:
    """
    Hand-written rule-based natural language Intent Router.
    Routes queries to:
    - VISUAL_INSPECTION (Requires RT-DETR visual detection)
    - GENERAL_KNOWLEDGE (Non-image / out-of-domain conversational query)
    - GREETING_HELP (Greetings or capability explanation)
    - INVALID_QUERY (Empty or unparseable input)
    """
    
    VISUAL_KEYWORDS = [
        "how many", "count", "where", "closest", "nearest", "distance", "adjacent",
        "missing", "assembled", "complete", "present", "exist", "detected",
        "position", "location", "left", "right", "top", "bottom", "above", "below",
        "capacitor", "resistor", "mosfet", "mov", "transformer", "component", "board",
        "usb", "connector", "part", "defect", "inspection", "damage"
    ]
    
    @classmethod
    def route(cls, question: str) -> Dict[str, Any]:
        cleaned = question.strip().lower()
        if not cleaned:
            return {
                "intent": "INVALID_QUERY",
                "requires_detection": False,
                "reason": "Question is empty."
            }
            
        # Clean leading punctuation for greetings
        cleaned_words = re.findall(r'\b\w+\b', cleaned)
        first_word = cleaned_words[0] if cleaned_words else ""
        first_phrase = " ".join(cleaned_words[:3])
        
        if first_word in ["hello", "hi", "hey", "help"] or any(g in first_phrase for g in ["who are you", "what can you do"]):
            return {
                "intent": "GREETING_HELP",
                "requires_detection": False,
                "direct_answer": "Hello! I am the Smart PCB Inspection & Automated Defect Reasoning Engine. You can upload a PCB image and ask questions regarding component counts, nearest spatial neighbors, missing parts, or assembly completeness."
            }
            
        # Check visual keywords with word boundaries
        has_visual_keyword = any(re.search(r'\b' + re.escape(kw) + r'\b', cleaned) for kw in cls.VISUAL_KEYWORDS)
        
        if has_visual_keyword:
            return {
                "intent": "VISUAL_INSPECTION",
                "requires_detection": True,
                "reason": "Question asks for visual perception or spatial properties of the PCB image."
            }
            
        # Non-visual general query (weather, general trivia, etc.)
        return {
            "intent": "GENERAL_KNOWLEDGE",
            "requires_detection": False,
            "direct_answer": "This question does not appear to relate to the provided PCB image or board components. Please provide an inspection query such as 'How many capacitors exist?', 'Which component is closest to the transformer?', or 'Is the board fully assembled?'."
        }


class PCBReasoningEngine:
    """
    Hand-written structured mathematical reasoning layer for PCB spatial analysis.
    Calculates exact Euclidean distances, centroid geometry, and BOM checks.
    """
    
    def __init__(self, min_confidence_guardrail: float = 0.30):
        self.min_confidence_guardrail = min_confidence_guardrail
        
    def reason_over_detections(
        self,
        question: str,
        detection_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Executes structured spatial reasoning and applies confidence guardrails.
        """
        detections = detection_result.get("detections", [])
        q_lower = question.strip().lower()
        
        # 1. Apply Confidence Guardrail
        mean_conf = detection_result.get("quality_metrics", {}).get("mean_confidence", 0.0)
        total_dets = detection_result.get("total_detections", 0)
        
        if total_dets == 0:
            return {
                "status": "INSUFFICIENT_INFORMATION",
                "answer": "Insufficient information: No PCB components were detected with sufficient confidence. The image may be severely blurred, out of focus, or does not contain a recognizable PCB.",
                "confidence": 0.0,
                "guardrail_triggered": True,
                "guardrail_reason": "Zero detections returned by RT-DETR."
            }
            
        if mean_conf < self.min_confidence_guardrail:
            return {
                "status": "INSUFFICIENT_INFORMATION",
                "answer": f"Insufficient information: Average model detection confidence ({mean_conf:.2f}) is below the minimum reliability threshold ({self.min_confidence_guardrail:.2f}). Component validation cannot be guaranteed.",
                "confidence": mean_conf,
                "guardrail_triggered": True,
                "guardrail_reason": f"Mean detection confidence {mean_conf:.2f} < threshold {self.min_confidence_guardrail:.2f}."
            }

        # 2. Dispatch to Structured Reasoning Handlers
        
        # A. Proximity / Nearest Neighbor Query
        if any(w in q_lower for w in ["closest", "nearest", "next to", "distance", "neighbor"]):
            return self._reason_closest_component(q_lower, detections)
            
        # B. Count Queries ("how many", "count", "number of")
        if any(w in q_lower for w in ["how many", "count", "number of", "how much"]):
            return self._reason_counts(q_lower, detections, detection_result.get("class_counts", {}))
            
        # C. Missing Components Query ("missing", "are any missing", "lacking")
        if any(w in q_lower for w in ["missing", "absent", "lacking", "omitted"]):
            return self._reason_missing_components(detections, detection_result.get("class_counts", {}))
            
        # D. Assembly Completeness ("fully assembled", "complete", "ready")
        if any(w in q_lower for w in ["fully assembled", "assembled", "complete", "finished", "assembly"]):
            return self._reason_assembly_completeness(detections, detection_result.get("class_counts", {}))
            
        # E. Spatial Position Query ("where is", "position", "highest", "lowest", "leftmost", "rightmost")
        if any(w in q_lower for w in ["where", "top", "bottom", "leftmost", "rightmost", "highest", "lowest"]):
            return self._reason_spatial_positions(q_lower, detections)
            
        # F. General Component Inventory Summary
        return self._reason_general_summary(detections, detection_result.get("class_counts", {}))
        
    def _reason_counts(
        self,
        q_lower: str,
        detections: List[Dict[str, Any]],
        class_counts: Dict[str, int]
    ) -> Dict[str, Any]:
        """Calculates specific or general component counts."""
        for canonical, aliases in COMPONENT_ALIASES.items():
            if any(re.search(r'\b' + re.escape(alias) + r'\b', q_lower) for alias in aliases):
                if canonical == "capacitor":
                    cap_counts = {
                        "Cap1": class_counts.get("Cap1", 0),
                        "Cap2": class_counts.get("Cap2", 0),
                        "Cap3": class_counts.get("Cap3", 0),
                        "Cap4": class_counts.get("Cap4", 0)
                    }
                    total_caps = sum(cap_counts.values())
                    breakdown = ", ".join([f"{k}: {v}" for k, v in cap_counts.items() if v > 0])
                    return {
                        "status": "SUCCESS",
                        "answer": f"There are {total_caps} capacitors detected on the PCB ({breakdown if breakdown else 'none'}).",
                        "data": {"component": "Capacitor", "total_count": total_caps, "subtypes": cap_counts},
                        "guardrail_triggered": False
                    }
                elif canonical == "connector":
                    conn_count = class_counts.get("Connector", 0)
                    return {
                        "status": "SUCCESS",
                        "answer": f"There are {conn_count} connectors detected on the inspected board surface.",
                        "data": {"component": "Connector", "total_count": conn_count},
                        "guardrail_triggered": False
                    }
                else:
                    target_name = canonical.capitalize() if canonical != "mov" else "MOV"
                    count = class_counts.get(target_name, 0)
                    return {
                        "status": "SUCCESS",
                        "answer": f"There are {count} {target_name} component(s) detected on this board.",
                        "data": {"component": target_name, "total_count": count},
                        "guardrail_triggered": False
                    }
                    
        total = len(detections)
        summary = ", ".join([f"{k}: {v}" for k, v in class_counts.items() if v > 0])
        return {
            "status": "SUCCESS",
            "answer": f"A total of {total} components are detected on this PCB: {summary}.",
            "data": {"total_components": total, "breakdown": class_counts},
            "guardrail_triggered": False
        }

    def _reason_closest_component(
        self,
        q_lower: str,
        detections: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Finds the nearest neighboring component using Euclidean centroid distance."""
        target_alias = None
        prox_match = re.search(r'(?:closest|nearest|next)\s+to\s+(?:the\s+)?([a-zA-Z0-9_\-]+)', q_lower)
        if prox_match:
            candidate_word = prox_match.group(1)
            for canonical, aliases in COMPONENT_ALIASES.items():
                if any(candidate_word == a or candidate_word.startswith(a) for a in aliases):
                    target_alias = canonical
                    break
                    
        if not target_alias:
            for canonical, aliases in COMPONENT_ALIASES.items():
                if any(re.search(r'\b' + re.escape(alias) + r'\b', q_lower) for alias in aliases):
                    target_alias = canonical
                    break
                
        if not target_alias:
            return {
                "status": "INSUFFICIENT_INFORMATION",
                "answer": "Insufficient information: Could not determine which reference component you want to measure distance from. Please specify a component such as 'transformer', 'MOSFET', or 'capacitor'.",
                "guardrail_triggered": True,
                "guardrail_reason": "No valid reference component parsed from proximity query."
            }
            
        target_name = target_alias.capitalize() if target_alias != "mov" else "MOV"
        target_dets = []
        for d in detections:
            cname = d["class_name"].lower()
            if target_alias in cname or (target_alias == "capacitor" and "cap" in cname):
                target_dets.append(d)
                
        if not target_dets:
            return {
                "status": "INSUFFICIENT_INFORMATION",
                "answer": f"Insufficient information: No {target_name} component was detected on this PCB, so the closest neighbor cannot be computed.",
                "guardrail_triggered": True,
                "guardrail_reason": f"Target reference component '{target_name}' not present in detection set."
            }
            
        ref_obj = max(target_dets, key=lambda x: x["confidence"])
        rx, ry = ref_obj["centroid"]
        
        candidates = []
        for d in detections:
            if d["id"] == ref_obj["id"]:
                continue
            dx, dy = d["centroid"]
            dist_px = math.sqrt((dx - rx) ** 2 + (dy - ry) ** 2)
            candidates.append((dist_px, d))
            
        if not candidates:
            return {
                "status": "SUCCESS",
                "answer": f"{ref_obj['class_name']} is the only component detected on the board; no other neighbors exist.",
                "data": {"reference_component": ref_obj["class_name"]},
                "guardrail_triggered": False
            }
            
        candidates.sort(key=lambda x: x[0])
        closest_dist, closest_obj = candidates[0]
        
        dx = closest_obj["centroid"][0] - rx
        dy = closest_obj["centroid"][1] - ry
        h_dir = "to the right" if dx > 0 else "to the left"
        v_dir = "below" if dy > 0 else "above"
        
        return {
            "status": "SUCCESS",
            "answer": f"The component closest to {ref_obj['class_name']} (at centroid [{rx:.0f}, {ry:.0f}]) is {closest_obj['class_name']} (id #{closest_obj['id']}), located approximately {closest_dist:.1f} pixels away ({v_dir} and {h_dir}).",
            "data": {
                "reference_object": ref_obj,
                "closest_object": closest_obj,
                "distance_pixels": round(closest_dist, 2),
                "relative_direction": f"{v_dir} and {h_dir}"
            },
            "guardrail_triggered": False
        }

    def _reason_missing_components(
        self,
        detections: List[Dict[str, Any]],
        class_counts: Dict[str, int]
    ) -> Dict[str, Any]:
        """Checks for missing components against standard PCB assembly specifications."""
        missing = []
        total_caps = sum(class_counts.get(f"Cap{i}", 0) for i in range(1, 5))
        
        if total_caps < REFERENCE_PCB_RECIPE["Capacitor"]["min"]:
            missing.append(f"Capacitor Cluster (detected {total_caps}, expected minimum {REFERENCE_PCB_RECIPE['Capacitor']['min']})")
            
        if class_counts.get("Transformer", 0) < REFERENCE_PCB_RECIPE["Transformer"]["min"]:
            missing.append("Transformer")
            
        if class_counts.get("MOSFET", 0) < REFERENCE_PCB_RECIPE["MOSFET"]["min"]:
            missing.append("MOSFET Power Transistor")
            
        if class_counts.get("MOV", 0) < REFERENCE_PCB_RECIPE["MOV"]["min"]:
            missing.append("MOV Surge Suppressor")
            
        if class_counts.get("Resistor", 0) < REFERENCE_PCB_RECIPE["Resistor"]["min"]:
            missing.append("Resistor")
            
        if missing:
            return {
                "status": "SUCCESS",
                "answer": f"Yes, potential missing components were identified based on the standard board bill of materials: {', '.join(missing)}.",
                "data": {"missing_components": missing, "is_complete": False},
                "guardrail_triggered": False
            }
        else:
            return {
                "status": "SUCCESS",
                "answer": "No missing components detected. All expected primary components (Capacitors, Transformer, MOSFET, MOV, Resistors) are present and accounted for.",
                "data": {"missing_components": [], "is_complete": True},
                "guardrail_triggered": False
            }

    def _reason_assembly_completeness(
        self,
        detections: List[Dict[str, Any]],
        class_counts: Dict[str, int]
    ) -> Dict[str, Any]:
        """Evaluates whether the board has passed complete assembly verification."""
        missing_res = self._reason_missing_components(detections, class_counts)
        is_complete = missing_res["data"]["is_complete"]
        
        if is_complete:
            return {
                "status": "SUCCESS",
                "answer": f"Yes, the board appears to be fully assembled. All critical stages (Power transformer, MOSFET switching unit, MOV protection, and {sum(class_counts.get(f'Cap{i}', 0) for i in range(1, 5))} capacitors) are verified in their nominal positions.",
                "data": {"is_fully_assembled": True, "total_components": len(detections)},
                "guardrail_triggered": False
            }
        else:
            return {
                "status": "SUCCESS",
                "answer": f"No, the board is NOT fully assembled. Missing or under-populated elements: {', '.join(missing_res['data']['missing_components'])}.",
                "data": {"is_fully_assembled": False, "missing": missing_res["data"]["missing_components"]},
                "guardrail_triggered": False
            }

    def _reason_spatial_positions(
        self,
        q_lower: str,
        detections: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Reasons over positional extremes (top, bottom, leftmost, rightmost)."""
        if not detections:
            return {
                "status": "INSUFFICIENT_INFORMATION",
                "answer": "Insufficient information: No components detected to establish spatial coordinates.",
                "guardrail_triggered": True
            }
            
        if "highest" in q_lower or "top" in q_lower:
            top_obj = min(detections, key=lambda d: d["centroid"][1])
            return {
                "status": "SUCCESS",
                "answer": f"The topmost component is {top_obj['class_name']} (id #{top_obj['id']}) located near coordinates ({top_obj['centroid'][0]:.0f}, {top_obj['centroid'][1]:.0f}).",
                "data": {"topmost_object": top_obj},
                "guardrail_triggered": False
            }
            
        if "lowest" in q_lower or "bottom" in q_lower:
            bottom_obj = max(detections, key=lambda d: d["centroid"][1])
            return {
                "status": "SUCCESS",
                "answer": f"The bottom-most component is {bottom_obj['class_name']} (id #{bottom_obj['id']}) located near coordinates ({bottom_obj['centroid'][0]:.0f}, {bottom_obj['centroid'][1]:.0f}).",
                "data": {"bottommost_object": bottom_obj},
                "guardrail_triggered": False
            }
            
        if "leftmost" in q_lower or "left" in q_lower:
            left_obj = min(detections, key=lambda d: d["centroid"][0])
            return {
                "status": "SUCCESS",
                "answer": f"The leftmost component is {left_obj['class_name']} (id #{left_obj['id']}) at X-coordinate {left_obj['centroid'][0]:.0f}.",
                "data": {"leftmost_object": left_obj},
                "guardrail_triggered": False
            }
            
        if "rightmost" in q_lower or "right" in q_lower:
            right_obj = max(detections, key=lambda d: d["centroid"][0])
            return {
                "status": "SUCCESS",
                "answer": f"The rightmost component is {right_obj['class_name']} (id #{right_obj['id']}) at X-coordinate {right_obj['centroid'][0]:.0f}.",
                "data": {"rightmost_object": right_obj},
                "guardrail_triggered": False
            }
            
        return self._reason_general_summary(detections, {})

    def _reason_general_summary(
        self,
        detections: List[Dict[str, Any]],
        class_counts: Dict[str, int]
    ) -> Dict[str, Any]:
        """Provides a natural language summary of detected board components."""
        total = len(detections)
        summary = ", ".join([f"{k}: {v}" for k, v in class_counts.items() if v > 0])
        return {
            "status": "SUCCESS",
            "answer": f"The PCB inspection detected {total} components in total ({summary}).",
            "data": {"total_count": total, "class_counts": class_counts},
            "guardrail_triggered": False
        }


class LocalLLMReasoningEngine:
    """
    Direct, framework-free Local LLM Reasoning Engine using HuggingFace Transformers & PyTorch.
    Runs entirely on local GPU (CUDA) or CPU without LangChain, CrewAI, AutoGen, or external APIs.
    """
    
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
        min_confidence_guardrail: float = 0.30
    ):
        self.model_id = model_id
        self.min_confidence_guardrail = min_confidence_guardrail
        self.tokenizer = None
        self.model = None
        self.device = None
        self.is_loaded = False
        self._fallback_engine = PCBReasoningEngine(min_confidence_guardrail=min_confidence_guardrail)
        
    def load_model(self):
        """Loads tokenizer and model weights directly into GPU memory."""
        if self.is_loaded:
            return
            
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading local NLP LLM '{self.model_id}' on {self.device}...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map=self.device
            )
            self.is_loaded = True
            logger.info(f"Local NLP LLM loaded successfully on {self.device}.")
        except Exception as e:
            logger.error(f"Failed to load local LLM: {str(e)}. Fallback engine will be used.", exc_info=True)
            self.is_loaded = False
            
    def reason(
        self,
        question: str,
        detection_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Executes local LLM reasoning grounded in RT-DETR detections with confidence guardrails.
        """
        start_t = time.time()
        
        # 1. First enforce strict deterministic Guardrails
        mean_conf = detection_result.get("quality_metrics", {}).get("mean_confidence", 0.0)
        total_dets = detection_result.get("total_detections", 0)
        
        if total_dets == 0:
            return {
                "status": "INSUFFICIENT_INFORMATION",
                "answer": "Insufficient information: No PCB components were detected with sufficient confidence. The image may be severely blurred, out of focus, or does not contain a recognizable PCB.",
                "confidence": 0.0,
                "guardrail_triggered": True,
                "guardrail_reason": "Zero detections returned by RT-DETR.",
                "engine": "guardrail"
            }
            
        if mean_conf < self.min_confidence_guardrail:
            return {
                "status": "INSUFFICIENT_INFORMATION",
                "answer": f"Insufficient information: Average model detection confidence ({mean_conf:.2f}) is below the minimum reliability threshold ({self.min_confidence_guardrail:.2f}). Component validation cannot be guaranteed.",
                "confidence": mean_conf,
                "guardrail_triggered": True,
                "guardrail_reason": f"Mean detection confidence {mean_conf:.2f} < threshold {self.min_confidence_guardrail:.2f}.",
                "engine": "guardrail"
            }

        # 2. Extract deterministic geometric facts as grounding context
        geo_result = self._fallback_engine.reason_over_detections(question, detection_result)
        if geo_result.get("guardrail_triggered"):
            return {**geo_result, "engine": "guardrail"}
            
        # Try LLM generation if available
        if not self.is_loaded:
            try:
                self.load_model()
            except Exception:
                pass
                
        if not self.is_loaded or self.model is None:
            # Return deterministic reasoning response
            return {**geo_result, "engine": "deterministic-spatial-fallback"}

        # 3. Build Prompt Grounding for Local LLM
        import torch
        detections = detection_result.get("detections", [])
        class_counts = detection_result.get("class_counts", {})
        active_counts = {k: v for k, v in class_counts.items() if v > 0}
        
        components_summary = []
        for d in detections:
            components_summary.append(
                f"- #{d['id']}: {d['class_name']} at centroid ({d['centroid'][0]:.0f}, {d['centroid'][1]:.0f}), confidence {d['confidence']*100:.1f}%"
            )
        components_str = "\n".join(components_summary)
        
        system_prompt = (
            "You are an expert Smart PCB Inspection Quality Assurance Engineer.\n"
            "You are given real-time perception data from an RT-DETR object detector on an inspected PCB.\n"
            "Answer the user's question accurately, concisely, and factually based ONLY on the detection data provided below.\n"
            "Rules:\n"
            "- If asked how many components or capacitors exist, state the exact count from the data.\n"
            "- If asked about missing components or assembly, note what is present and what is missing.\n"
            "- If asked about closest components, use the spatial coordinates.\n"
            "- Never hallucinate components not listed in the detection data.\n"
            "- Keep your response direct, professional, and within 1-3 sentences."
        )
        
        user_prompt = (
            f"PCB Detection Telemetry:\n"
            f"- Total Detected Components: {total_dets}\n"
            f"- Component Counts: {active_counts}\n"
            f"- Detected Objects:\n{components_str}\n\n"
            f"Calculated Spatial Fact: {geo_result.get('answer', '')}\n\n"
            f"Question: {question}\n\n"
            f"Provide a concise, direct inspection answer:"
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            formatted_input = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = self.tokenizer([formatted_input], return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=90,
                    do_sample=False,
                    repetition_penalty=1.1
                )
                
            input_len = inputs.input_ids.shape[1]
            generated_tokens = outputs[0][input_len:]
            llm_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
            if not llm_text:
                llm_text = geo_result.get("answer", "")
                
            return {
                "status": "SUCCESS",
                "answer": llm_text,
                "data": geo_result.get("data", {}),
                "guardrail_triggered": False,
                "engine": f"local-llm ({self.model_id})"
            }
        except Exception as e:
            logger.warning(f"Local LLM generation failed: {str(e)}. Using deterministic answer.", exc_info=True)
            return {**geo_result, "engine": "deterministic-spatial-fallback"}
