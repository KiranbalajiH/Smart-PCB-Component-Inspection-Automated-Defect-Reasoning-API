# Smart PCB Inspection & Defect Reasoning System
**Pre-Hackathon Screening • Track: Computer Vision + Applied ML Engineering**

---

## 1. Executive Summary & Domain Choice

### Domain & Dataset Sourcing
We selected **Automated Optical Inspection (AOI) for Printed Circuit Board (PCB) Manufacturing**. Modern SMT (Surface Mount Technology) assembly lines require real-time, zero-defect quality control to detect missing, misaligned, or incorrect electronic components before solder reflow.

All target classes are strictly **non-COCO industrial electronics classes**:
- **Capacitors**: Granular multi-class electrolytic and ceramic capacitors (`Cap1`, `Cap2`, `Cap3`, `Cap4`).
- **MOSFETs**: Power semiconductor switching transistors.
- **MOV (Metal Oxide Varistor)**: Over-voltage surge protection components.
- **Transformers**: High-frequency ferrite power induction coils.
- **Resistors**: Standard surface-mount / through-hole resistors (including automated normalization of raw dataset labeling typos like `Resestor`).

The raw dataset was sourced from high-resolution PCB optical inspection captures, containing 1,296 fully labeled board images with 11,119 component bounding boxes.

```
       [Raw Optical PCB Stream]
                  │
                  ▼
      ┌────────────────────────┐
      │  RT-DETR-L Perception  │ ──► [Boxes, Classes, Confidences, Centroids]
      └────────────────────────┘                      │
                  │                                   ▼
                  │                    ┌───────────────────────────────┐
                  ▼                    │ Hand-Written Reasoning Engine │
      ┌────────────────────────┐       ├───────────────────────────────┤
      │  FastAPI /detect API   │       │ • Intent Routing (Regex/NLP)  │
      └────────────────────────┘       │ • Nearest-Neighbor Proximity  │
                                       │ • Missing BOM Verification    │
                                       │ • Confidence Guardrails       │
                                       └───────────────────────────────┘
                                                      │
                                                      ▼
                                       ┌───────────────────────────────┐
                                       │  FastAPI /reason Output & UI  │
                                       └───────────────────────────────┘
```

---

## 2. Train / Val / Test Split Strategy & Technical Justification

### Split Methodology
- **Total Gold-Annotated Dataset**: 1,296 images, 11,119 bounding box annotations.
- **Partition Ratio**: 80% Train (1,036 images, 8,894 instances) | 10% Validation (129 images, 1,098 instances) | 10% Holdout Test (131 images, 1,127 instances).
- **Random Seed**: Fixed at `42` with deterministic coordinate normalization for full reproducibility.

### Technical Justification
In the raw public dataset download, the `test` directory contained unannotated images (`objects: []`). Reporting metrics against unannotated images produces artificial zero scores and precludes diagnostic failure analysis. To ensure genuine, rigorous evaluation, we pooled all verified gold-standard samples and partitioned them into an independent holdout test set (131 images) that the model never saw during training or validation hyperparameter tuning.

---

## 3. Evaluation Metrics: Insights, Limits & Manufacturing Trade-offs

### Quantitative Metrics on 131-Image Holdout Test Set
| Evaluation Metric | Test Set Score | Industrial Interpretation |
| :--- | :---: | :--- |
| **mAP@0.50** | **31.80%** | Baseline multi-class localization at nominal bounding box overlap (IoU ≥ 0.50). |
| **mAP@0.50:0.95** | **20.30%** | Strict boundary precision across rigorous IoU thresholds (0.50 to 0.95). |
| **Overall Recall** | **63.33%** | Overall component detection rate across all 8 classes. |
| **Cap1 (Capacitor) Recall** | **100.0%** | Zero false negatives for primary electrolytic capacitors. |
| **MOSFET Recall** | **99.24%** | Critical switching transistor recall; catches 99.2%+ of populated ICs. |
| **Resistor Recall** | **90.83%** | Strong detection of small SMD and through-hole resistors. |
| **MOV Recall** | **77.10%** | Reliable surge suppressor localization. |
| **Inference Latency (GPU)** | **31.5 ms** | Real-time high-throughput inspection (~31 FPS on RTX 4050). |

### What These Metrics Tell Us
1. **Recall Prioritization in SMT Production**: In PCB manufacturing, **Recall is heavily prioritized over Precision**. A false positive (flagging a good component) only triggers a 2-second operator verification, whereas a false negative (missing an unpopulated capacitor, missing resistor, or unmounted MOV) allows a defective power board into production, causing catastrophic field failure.
2. **Capacitor Sub-type Breakdown**: While broad components (`Cap1`, `MOSFET`, `Resistor`) achieved 90%–100% recall, granular inter-capacitor subtypes (`Cap2`, `Cap3`, `Cap4`) share near-identical silver can dimensions, leading to subtype confusion that our Part B reasoning layer addresses.

### What mAP Fails to Tell Us
- **Topological & Connectivity Correctness**: mAP evaluates box overlap in isolation. It cannot verify whether a detected capacitor is soldered to the correct polarity trace or if pin-1 clearance tolerances are respected. This necessitates our **Part B Spatial Reasoning Layer**.

---

## 4. Root-Cause Analysis of Five Real Model Failure Cases

Honest error acknowledgment is paramount. We extracted 5 distinct failure modes from our test set:

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Case 1: Small SMD Resistor Miss (False Negative)                                                  │
│ Root Cause: Receptive field downsampling (32x stride) causes sub-15px SMD resistors to lose        │
│ gradient contrast against green solder mask traces.                                                │
├────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Case 2: Inter-Capacitor Subtype Confusion (Cap1 vs. Cap2/Cap3)                                     │
│ Root Cause: Radial electrolytic cylindrical cans share identical circular top geometries and silver│
│ vent markings; diameter variation is minimal without calibrated depth perception.                  │
├────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Case 3: Solder Fillet Specular Reflection (False Positive)                                         │
│ Root Cause: High-intensity directional LED lighting creates localized hot spots on reflow solder    │
│ pads, occasionally mimicking the rectangular reflective package of a chip resistor.              │
├────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Case 4: Extreme Camera Angle & Perspective Distortion                                              │
│ Root Cause: Oblique viewpoint angles (>35° off-nadir) compress vertical component aspect ratios,   │
│ causing edge boundary blurring and lowering confidence scores below 0.35.                          │
├────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Case 5: Transformer Overhang Boundary Occlusion                                                    │
│ Root Cause: Tall ferrite transformer cores cast physical shadows and partially occlude adjacent   │
│ low-profile ceramic capacitors positioned within 2mm of the core base.                             │
└────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Part B: Hand-Written Reasoning Layer & Confidence Guardrails

In strict compliance with screening rules, Part B is implemented in **pure Python without any agentic frameworks** (zero LangChain, LangGraph, CrewAI, or AutoGen).

### Architecture & Intent Routing
1. **Intent Router**: Tokenizes queries and applies regex word-boundary matching.
   - Non-visual questions (*"What is the weather today?"*, *"Hello"*) are routed to direct responses without wasting GPU perception cycles.
   - Visual inspection questions (*"How many capacitors exist?"*, *"Which component is closest to the transformer?"*, *"Is the board fully assembled?"*) trigger RT-DETR detection and structured spatial reasoning.
2. **Geometric Reasoning Engine**:
   - **Euclidean Nearest-Neighbor**: Computes $d = \sqrt{(x_2-x_1)^2 + (y_2-y_1)^2}$ across all detected component centroids and outputs relative direction (*"MOSFET is 28.3px below and to the right of Transformer"*).
   - **Bill of Materials (BOM) Verification**: Validates detected component counts against the nominal reference PCB recipe.
3. **Confidence Guardrail & Explicit "Insufficient Information" Output**:
   - If mean detection confidence falls below $0.30$, or if the requested reference entity is not found, the engine explicitly refuses to guess:
   - **Example**: When evaluated on an out-of-focus image where mean confidence dropped to $0.21$:
     > `"Insufficient information: Average model detection confidence (0.21) is below the minimum reliability threshold (0.30). Component validation cannot be guaranteed."`

---

## 6. API Usage Guide & Verification Payloads

### Starting the Server
```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000
# Or using Docker:
docker-compose up --build
```

### 1. Object Detection (`POST /detect`)
```bash
curl -X POST "http://localhost:8000/detect" \
     -F "file=@data/pcb_yolo/images/test/VID20210601143927-96_jpg.rf.51037a593015ac38c8ac31613da3c066.jpg" \
     -F "conf=0.35"
```
**Sample JSON Response:**
```json
{
  "status": "success",
  "inference_time_ms": 14.8,
  "total_detections": 8,
  "class_counts": { "Cap1": 2, "Cap2": 1, "Cap3": 1, "Cap4": 1, "Transformer": 1, "MOSFET": 1, "MOV": 1 },
  "quality_metrics": { "mean_confidence": 0.912, "min_confidence": 0.745 }
}
```

### 2. Natural Language Reasoning (`POST /reason`)
```bash
curl -X POST "http://localhost:8000/reason" \
     -F "file=@data/pcb_yolo/images/test/VID20210601143927-96_jpg.rf.51037a593015ac38c8ac31613da3c066.jpg" \
     -F "question=Which component is closest to the transformer?"
```
**Sample JSON Response:**
```json
{
  "status": "SUCCESS",
  "intent": "VISUAL_INSPECTION",
  "requires_detection": true,
  "answer": "The component closest to Transformer (at centroid [540, 810]) is MOSFET (id #3), located approximately 42.1 pixels away (below and to the right).",
  "confidence_guardrail_triggered": false,
  "inference_time_ms": 16.2
}
```

---
*Developed for Round 1 CV + Applied ML Pre-Hackathon Screening.*
