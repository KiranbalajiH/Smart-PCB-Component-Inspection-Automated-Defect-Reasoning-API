let currentImageFile = null;
let currentImageBitmap = null;
let latestDetections = [];

const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");
const uploadPrompt = document.getElementById("uploadPrompt");
const canvasWrapper = document.getElementById("canvasWrapper");
const canvas = document.getElementById("inspectionCanvas");
const ctx = canvas.getContext("2d");

const confRange = document.getElementById("confRange");
const confValue = document.getElementById("confValue");
const btnRunDetect = document.getElementById("btnRunDetect");
const questionInput = document.getElementById("questionInput");
const btnAskReason = document.getElementById("btnAskReason");
const systemStatus = document.getElementById("systemStatus");

const CLASS_COLORS = {
    "Cap1": "#ef4444",
    "Cap2": "#f59e0b",
    "Cap3": "#3b82f6",
    "Cap4": "#8b5cf6",
    "MOSFET": "#ec4899",
    "MOV": "#06b6d4",
    "Resistor": "#10b981",
    "Transformer": "#eab308"
};

// Check health on startup
async function checkHealth() {
    try {
        const res = await fetch("/health");
        const data = await res.json();
        if (data.status === "healthy") {
            systemStatus.textContent = data.model_weights_loaded ? "Model Ready (CUDA)" : "API Ready";
            systemStatus.style.borderColor = "rgba(16, 185, 129, 0.5)";
        }
    } catch (e) {
        systemStatus.textContent = "API Offline";
        systemStatus.style.color = "#f43f5e";
    }
}
checkHealth();

// Confidence slider
confRange.addEventListener("input", (e) => {
    confValue.textContent = parseFloat(e.target.value).toFixed(2);
});

// Drag and drop image
dropZone.addEventListener("click", () => fileInput.click());
dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) {
        handleFileSelect(e.dataTransfer.files[0]);
    }
});
fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
    }
});

function handleFileSelect(file) {
    if (!file.type.startsWith("image/")) {
        alert("Please upload an image file (JPG or PNG).");
        return;
    }
    currentImageFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        const img = new Image();
        img.onload = () => {
            currentImageBitmap = img;
            canvas.width = img.width;
            canvas.height = img.height;
            uploadPrompt.style.display = "none";
            canvasWrapper.style.display = "flex";
            renderCanvas();
        };
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);
}

function renderCanvas(highlightObj = null) {
    if (!currentImageBitmap) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(currentImageBitmap, 0, 0);

    // Draw detections
    latestDetections.forEach(det => {
        const [x1, y1, x2, y2] = det.box_xyxy;
        const color = CLASS_COLORS[det.class_name] || "#38bdf8";
        const isHighlight = highlightObj && highlightObj.id === det.id;

        ctx.strokeStyle = isHighlight ? "#ffffff" : color;
        ctx.lineWidth = isHighlight ? 5 : 3;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

        // Label background
        const label = `#${det.id} ${det.class_name} ${(det.confidence * 100).toFixed(0)}%`;
        ctx.font = "bold 16px 'JetBrains Mono', sans-serif";
        const textWidth = ctx.measureText(label).width;
        
        ctx.fillStyle = isHighlight ? "#ffffff" : color;
        ctx.fillRect(x1, Math.max(0, y1 - 24), textWidth + 10, 24);

        ctx.fillStyle = isHighlight ? "#000000" : "#ffffff";
        ctx.fillText(label, x1 + 5, Math.max(18, y1 - 6));

        // Draw centroid point
        ctx.fillStyle = isHighlight ? "#ffffff" : color;
        ctx.beginPath();
        ctx.arc(det.centroid[0], det.centroid[1], 4, 0, 2 * Math.PI);
        ctx.fill();
    });
}

// Run Detection
btnRunDetect.addEventListener("click", async () => {
    if (!currentImageFile) {
        alert("Please upload a PCB image first.");
        return;
    }

    btnRunDetect.disabled = true;
    btnRunDetect.textContent = "Processing...";

    const formData = new FormData();
    formData.append("file", currentImageFile);
    const conf = parseFloat(confRange.value);

    try {
        const res = await fetch(`/detect?conf=${conf}`, {
            method: "POST",
            body: formData
        });
        const data = await res.json();

        if (res.ok) {
            latestDetections = data.detections;
            renderCanvas();
            updateTelemetry(data);
        } else {
            alert(`Detection error: ${data.detail}`);
        }
    } catch (e) {
        alert(`Failed to communicate with detection API: ${e.message}`);
    } finally {
        btnRunDetect.disabled = false;
        btnRunDetect.textContent = "🔍 Run Object Detection";
    }
});

function updateTelemetry(data) {
    document.getElementById("detCountBadge").textContent = `${data.total_detections} Detections`;
    document.getElementById("statInfTime").textContent = `${data.inference_time_ms} ms`;
    document.getElementById("statMeanConf").textContent = `${(data.quality_metrics.mean_confidence * 100).toFixed(1)}%`;
    
    const capCount = (data.class_counts.Cap1 || 0) + (data.class_counts.Cap2 || 0) + (data.class_counts.Cap3 || 0) + (data.class_counts.Cap4 || 0);
    document.getElementById("statCaps").textContent = `${capCount} units`;

    const powerCount = (data.class_counts.Transformer || 0) + (data.class_counts.MOSFET || 0) + (data.class_counts.MOV || 0);
    document.getElementById("statPower").textContent = `${powerCount} units`;

    // Populate table
    const tbody = document.getElementById("detTableBody");
    tbody.innerHTML = "";
    if (data.detections.length === 0) {
        tbody.innerHTML = `<tr><td colspan="5" class="empty-table">No components detected above threshold.</td></tr>`;
        return;
    }

    data.detections.forEach(det => {
        const tr = document.createElement("tr");
        tr.style.cursor = "pointer";
        tr.innerHTML = `
            <td>#${det.id}</td>
            <td><span style="color: ${CLASS_COLORS[det.class_name] || '#fff'}; font-weight:600;">${det.class_name}</span></td>
            <td>${(det.confidence * 100).toFixed(1)}%</td>
            <td>[${det.centroid[0].toFixed(0)}, ${det.centroid[1].toFixed(0)}]</td>
            <td>${det.dimensions[0].toFixed(0)} × ${det.dimensions[1].toFixed(0)}</td>
        `;
        tr.addEventListener("mouseenter", () => renderCanvas(det));
        tr.addEventListener("mouseleave", () => renderCanvas(null));
        tbody.appendChild(tr);
    });
}

// Run Part B Reasoning
btnAskReason.addEventListener("click", async () => {
    const question = questionInput.value.trim();
    if (!question) {
        alert("Please enter a question.");
        return;
    }

    btnAskReason.disabled = true;
    btnAskReason.textContent = "Reasoning...";

    const formData = new FormData();
    formData.append("question", question);
    if (currentImageFile) {
        formData.append("file", currentImageFile);
    }

    try {
        const res = await fetch("/reason", {
            method: "POST",
            body: formData
        });
        const data = await res.json();

        if (res.ok) {
            document.getElementById("intentBadge").textContent = data.intent;
            document.getElementById("reasonTime").textContent = `${data.inference_time_ms} ms`;
            document.getElementById("reasonAnswer").textContent = data.answer;

            const guardrailAlert = document.getElementById("guardrailAlert");
            if (data.confidence_guardrail_triggered) {
                guardrailAlert.style.display = "block";
                document.getElementById("guardrailReason").textContent = data.guardrail_reason || "Insufficient visual reliability.";
            } else {
                guardrailAlert.style.display = "none";
            }

            // Highlight nearest object if present in data
            if (data.data && data.data.closest_object) {
                renderCanvas(data.data.closest_object);
            }
        } else {
            alert(`Reasoning error: ${data.detail}`);
        }
    } catch (e) {
        alert(`Failed to query reasoning endpoint: ${e.message}`);
    } finally {
        btnAskReason.disabled = false;
        btnAskReason.textContent = "💬 Ask Question";
    }
});

// Sample Query Pills
document.querySelectorAll(".sample-btn").forEach(btn => {
    btn.addEventListener("click", () => {
        questionInput.value = btn.getAttribute("data-q");
        btnAskReason.click();
    });
});
