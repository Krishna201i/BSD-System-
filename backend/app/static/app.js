const form = document.getElementById("prediction-form");
const resultCard = document.getElementById("result-card");
const progressBar = document.getElementById("form-progress-bar");
const submitButton = document.getElementById("submit-btn");
const submitText = document.getElementById("submit-text");
const submitLoading = document.getElementById("submit-loading");
const riskPercentageEl = document.getElementById("risk-percentage");
const riskBadgeEl = document.getElementById("risk-badge");
const riskGaugeEl = document.getElementById("risk-gauge");
const riskMessageEl = document.getElementById("risk-message");
const factorListEl = document.getElementById("factor-list");
const tipsListEl = document.getElementById("tips-list");
const reportDropzoneEl = document.getElementById("report-dropzone");
const reportFileInputEl = document.getElementById("report-file-input");
const reportFileNameEl = document.getElementById("report-file-name");
const analyzeReportBtn = document.getElementById("analyze-report-btn");
const analyzeReportText = document.getElementById("analyze-report-text");
const analyzeReportLoading = document.getElementById("analyze-report-loading");
const reportSuccessBanner = document.getElementById("report-success-banner");
const reportSummaryEl = document.getElementById("report-summary");

// ── Model Stats Banner ──────────────────────────────────────────────────────
const modelStatsNameEl = document.getElementById("model-stats-name");
const statAccuracyEl   = document.getElementById("stat-accuracy-val");
const statRocEl        = document.getElementById("stat-roc-val");
const statF1El         = document.getElementById("stat-f1-val");

function animateCounter(el, target, suffix, decimals = 1, durationMs = 900) {
  if (!el) return;
  const start = performance.now();
  function step(now) {
    const progress = Math.min((now - start) / durationMs, 1);
    const eased = 1 - Math.pow(1 - progress, 3); // ease-out cubic
    el.textContent = (eased * target).toFixed(decimals) + suffix;
    if (progress < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

async function fetchModelInfo() {
  try {
    const res = await fetch("/model-info");
    if (!res.ok) return;
    const data = await res.json();
    if (modelStatsNameEl) {
      modelStatsNameEl.textContent = `Active Model: ${data.model_name}`;
    }
    animateCounter(statAccuracyEl, (data.test_accuracy || 0) * 100, "%");
    animateCounter(statRocEl,      (data.roc_auc       || 0) * 100, "%");
    animateCounter(statF1El,       (data.test_f1       || 0) * 100, "%");
  } catch (_) {
    if (modelStatsNameEl) modelStatsNameEl.textContent = "Model offline";
  }
}

// Fetch on load (only if banner exists on this page)
if (modelStatsNameEl) fetchModelInfo();
// ─────────────────────────────────────────────────────────────────────────────


const requiredFields = [...document.querySelectorAll("[data-required='true']")];
const autofillFieldIds = ["age", "gender", "hypertension", "heart_disease", "avg_glucose_level", "bmi", "smoking_status"];

const reportFieldMeta = {
  age: { label: "Age", kind: "input" },
  gender: { label: "Gender", kind: "select" },
  hypertension: { label: "Hypertension", kind: "select" },
  heart_disease: { label: "Heart Disease", kind: "select" },
  avg_glucose_level: { label: "Average Glucose Level", kind: "input" },
  bmi: { label: "BMI", kind: "input" },
  smoking_status: { label: "Smoking Status", kind: "select" },
};

let selectedReportFile = null;

function updateFormProgress() {
  const completed = requiredFields.filter((field) => String(field.value).trim().length > 0).length;
  const progress = requiredFields.length === 0 ? 0 : Math.round((completed / requiredFields.length) * 100);
  progressBar.style.width = `${progress}%`;
}

function setSubmittingState(isSubmitting) {
  submitButton.disabled = isSubmitting;
  submitText.classList.toggle("hidden", isSubmitting);
  submitLoading.classList.toggle("hidden", !isSubmitting);
}

function setReportAnalyzingState(isAnalyzing) {
  analyzeReportBtn.disabled = isAnalyzing;
  analyzeReportText.classList.toggle("hidden", isAnalyzing);
  analyzeReportLoading.classList.toggle("hidden", !isAnalyzing);
  reportDropzoneEl.classList.toggle("busy", isAnalyzing);
}

function clearReportSummary() {
  reportSummaryEl.innerHTML = "";
  reportSummaryEl.classList.add("hidden");
}

function clearResultCard() {
  if (!resultCard) {
    return;
  }

  resultCard.classList.add("hidden");
  resultCard.classList.remove("show", "high", "low");
}

function normalizeDisplayValue(fieldId, value) {
  if (value === null || value === undefined || value === "") {
    return "";
  }

  if (fieldId === "age") {
    return String(Math.trunc(Number(value)));
  }

  if (fieldId === "avg_glucose_level" || fieldId === "bmi") {
    return String(Number(value));
  }

  return String(value);
}

function setSelectPlaceholder(control, text) {
  if (control.tagName !== "SELECT" || control.options.length === 0) {
    return;
  }

  const defaultLabel = control.dataset.defaultBlankLabel || control.options[0].textContent || "Select";
  control.dataset.defaultBlankLabel = defaultLabel;
  control.options[0].textContent = text || defaultLabel;
}

function clearFieldAutofillState(fieldId) {
  const control = document.getElementById(fieldId);
  const field = control.closest(".field");
  field.classList.remove("autofilled", "missing");

  if (control.tagName === "SELECT") {
    setSelectPlaceholder(control, null);
  } else {
    control.placeholder = control.dataset.defaultPlaceholder || control.placeholder || "";
  }
}

function markFieldMissing(fieldId) {
  const control = document.getElementById(fieldId);
  const field = control.closest(".field");
  field.classList.remove("autofilled");
  field.classList.add("missing");

  control.value = "";
  if (control.tagName === "SELECT") {
    setSelectPlaceholder(control, "Not found in report");
  } else {
    control.placeholder = "Not found in report";
  }
}

function markFieldAutofilled(fieldId, value) {
  const control = document.getElementById(fieldId);
  const field = control.closest(".field");
  field.classList.add("autofilled");
  field.classList.remove("missing");

  if (control.tagName === "SELECT") {
    setSelectPlaceholder(control, null);
  } else if (!control.dataset.defaultPlaceholder) {
    control.dataset.defaultPlaceholder = control.getAttribute("placeholder") || "";
  }

  control.value = normalizeDisplayValue(fieldId, value);
}

function fillFieldsFromReport(extractedFields) {
  autofillFieldIds.forEach((fieldId) => {
    if (Object.prototype.hasOwnProperty.call(extractedFields, fieldId) && extractedFields[fieldId] !== null && extractedFields[fieldId] !== undefined && String(extractedFields[fieldId]).trim() !== "") {
      markFieldAutofilled(fieldId, extractedFields[fieldId]);
    } else {
      markFieldMissing(fieldId);
    }
  });

  updateFormProgress();
}

function renderReportSummary(extractedFields, missingFields) {
  const extractedItems = [];
  const missingItems = [];

  autofillFieldIds.forEach((fieldId) => {
    const meta = reportFieldMeta[fieldId];
    const value = extractedFields[fieldId];

    if (value === null || value === undefined || String(value).trim() === "") {
      missingItems.push(meta.label);
      return;
    }

    extractedItems.push(`${meta.label}: ${normalizeDisplayValue(fieldId, value)}`);
  });

  reportSummaryEl.innerHTML = "";

  const grid = document.createElement("div");
  grid.className = "report-summary-grid";

  const extractedBlock = document.createElement("div");
  extractedBlock.className = "report-summary-block";
  const extractedHeading = document.createElement("h4");
  extractedHeading.textContent = "Extracted";
  const extractedList = document.createElement("ul");

  if (extractedItems.length === 0) {
    const emptyItem = document.createElement("li");
    emptyItem.textContent = "None";
    extractedList.appendChild(emptyItem);
  } else {
    extractedItems.forEach((item) => {
      const listItem = document.createElement("li");
      listItem.textContent = item;
      extractedList.appendChild(listItem);
    });
  }

  extractedBlock.appendChild(extractedHeading);
  extractedBlock.appendChild(extractedList);

  const missingBlock = document.createElement("div");
  missingBlock.className = "report-summary-block";
  const missingHeading = document.createElement("h4");
  missingHeading.textContent = "Missing";
  const missingList = document.createElement("ul");

  const missingValues = missingFields && missingFields.length > 0 ? missingFields : missingItems;
  if (missingValues.length === 0) {
    const emptyItem = document.createElement("li");
    emptyItem.textContent = "None";
    missingList.appendChild(emptyItem);
  } else {
    missingValues.forEach((item) => {
      const listItem = document.createElement("li");
      listItem.textContent = reportFieldMeta[item]?.label || item;
      missingList.appendChild(listItem);
    });
  }

  missingBlock.appendChild(missingHeading);
  missingBlock.appendChild(missingList);

  grid.appendChild(extractedBlock);
  grid.appendChild(missingBlock);
  reportSummaryEl.appendChild(grid);
  reportSummaryEl.classList.remove("hidden");
}

function setSelectedReportFile(file) {
  selectedReportFile = file || null;
  reportFileNameEl.textContent = selectedReportFile ? selectedReportFile.name : "No file selected";
}

function openFilePicker() {
  reportFileInputEl.click();
}

async function analyzeSelectedReport() {
  if (!selectedReportFile) {
    alert("Choose a PDF or image report first.");
    return;
  }

  clearResultCard();
  clearReportSummary();
  reportSuccessBanner.classList.add("hidden");
  setReportAnalyzingState(true);

  try {
    const formData = new FormData();
    formData.append("file", selectedReportFile);

    const response = await fetch("/analyze-report", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Report analysis failed.");
    }

    fillFieldsFromReport(data.extracted_fields || {});
    renderReportSummary(data.extracted_fields || {}, data.missing_fields || []);
    reportSuccessBanner.classList.remove("hidden");

      // Save extracted fields and prediction for Edit & Resubmit flow
      try {
        const payloadForPrefill = {
          extracted_fields: data.extracted_fields || {},
          prediction: data.prediction || null,
        };
        localStorage.setItem("neuroscan_report_prefill", JSON.stringify(payloadForPrefill));
      } catch (e) {
        // ignore localStorage failures (e.g., private mode)
      }

    if (data.prediction) {
      renderResult(data.prediction);
    }
  } catch (error) {
    alert(error.message);
  } finally {
    setReportAnalyzingState(false);
  }
}

function animateGauge(targetValue, color) {
  let current = 0;
  const totalFrames = 48;
  const increment = targetValue / totalFrames;

  riskGaugeEl.style.setProperty("--gauge-color", color);

  const timer = setInterval(() => {
    current += increment;
    if (current >= targetValue) {
      current = targetValue;
      clearInterval(timer);
    }

    riskGaugeEl.style.setProperty("--gauge-value", `${current.toFixed(2)}`);
    riskPercentageEl.textContent = `${current.toFixed(1)}%`;
  }, 14);
}

function getTips(isHighRisk) {
  if (isHighRisk) {
    return [
      "Speak with a licensed clinician as soon as possible for personalized risk review.",
      "Track blood pressure and glucose consistently and follow prescribed treatment plans.",
      "Reduce smoking and limit alcohol while increasing gentle daily movement.",
    ];
  }

  return [
    "Keep regular preventive checkups and repeat screening over time.",
    "Maintain healthy sleep, hydration, and a balanced low-sodium diet.",
    "Continue moderate physical activity and monitor blood pressure periodically.",
  ];
}

function renderFactors(factors) {
  factorListEl.innerHTML = "";

  if (!factors || factors.length === 0) {
    const empty = document.createElement("p");
    empty.textContent = "No significant contributing factors were returned for this prediction.";
    factorListEl.appendChild(empty);
    return;
  }

  const maxAbsContribution = Math.max(...factors.map((factor) => Math.abs(Number(factor.contribution) || 0)), 1);

  factors.forEach((factor) => {
    const contribution = Number(factor.contribution) || 0;
    const width = Math.max((Math.abs(contribution) / maxAbsContribution) * 100, 6);
    const higherRisk = contribution > 0;

    const item = document.createElement("div");
    item.className = "factor-item";

    const row = document.createElement("div");
    row.className = "factor-row";

    const name = document.createElement("span");
    name.className = "factor-name";
    name.textContent = factor.feature;

    const direction = document.createElement("span");
    direction.className = "factor-dir";
    direction.textContent = factor.direction;

    const track = document.createElement("div");
    track.className = "factor-track";

    const fill = document.createElement("div");
    fill.className = `factor-fill ${higherRisk ? "high" : "low"}`;

    row.appendChild(name);
    row.appendChild(direction);
    track.appendChild(fill);
    item.appendChild(row);
    item.appendChild(track);
    factorListEl.appendChild(item);

    requestAnimationFrame(() => {
      fill.style.width = `${width}%`;
    });
  });
}

function renderTips(isHighRisk) {
  tipsListEl.innerHTML = "";
  const tips = getTips(isHighRisk);
  tips.forEach((tip) => {
    const item = document.createElement("li");
    item.textContent = tip;
    tipsListEl.appendChild(item);
  });
}

function renderResult(data) {
  const isHighRisk = data.risk_label === "High Risk" || data.risk_label === "High";
  const riskPercent = Number(data.risk_percentage || data.risk_percent) || 0;
  // If the main `resultCard` exists (index.html), update it as before.
  if (resultCard) {
    resultCard.classList.remove("hidden", "high", "low");
    resultCard.classList.add("show", isHighRisk ? "high" : "low");

    riskBadgeEl.textContent = isHighRisk ? "High Risk" : "Low Risk";
    riskBadgeEl.className = `risk-badge ${isHighRisk ? "high" : "low"}`;

    riskMessageEl.textContent = isHighRisk
      ? "\u26A0\uFE0F Your profile indicates elevated stroke risk. Please consider consulting a healthcare professional soon."
      : "\u2705 Your profile indicates lower stroke risk at this time. Keep following healthy habits and regular checkups.";

    animateGauge(riskPercent, isHighRisk ? "#ef4444" : "#10b981");
    renderFactors(data.key_contributing_factors || data.top_factors || []);
    renderTips(isHighRisk);
    return;
  }

  // Fallback for the standalone report page which uses `prediction-card`.
  const predCard = document.getElementById("prediction-card");
  const localRiskBadge = document.getElementById("risk-badge");
  const localRiskMessage = document.getElementById("risk-message");
  const localRiskPercentage = document.getElementById("risk-percentage");

  if (predCard) {
    predCard.classList.remove("hidden", "high", "low");
    predCard.classList.add(isHighRisk ? "high" : "low");
  }

  if (localRiskBadge) {
    localRiskBadge.textContent = isHighRisk ? "High Risk" : "Low Risk";
    localRiskBadge.className = `risk-badge ${isHighRisk ? "high" : "low"}`;
  }

  if (localRiskMessage) {
    localRiskMessage.textContent = isHighRisk
      ? "\u26A0\uFE0F Your profile indicates elevated stroke risk. Please consider consulting a healthcare professional soon."
      : "\u2705 Your profile indicates lower stroke risk at this time. Keep following healthy habits and regular checkups.";
  }

  // Update gauge and factors (these elements exist on report.html)
  animateGauge(riskPercent, isHighRisk ? "#ef4444" : "#10b981");
  renderFactors(data.key_contributing_factors || data.top_factors || []);
  renderTips(isHighRisk);
}

requiredFields.forEach((field) => {
  field.addEventListener("input", updateFormProgress);
  field.addEventListener("change", updateFormProgress);
});

autofillFieldIds.forEach((fieldId) => {
  const control = document.getElementById(fieldId);
  if (!control) {
    return;
  }

  if (control.tagName === "INPUT") {
    control.dataset.defaultPlaceholder = control.getAttribute("placeholder") || "";
  }

  control.addEventListener("input", () => {
    clearFieldAutofillState(fieldId);
    updateFormProgress();
  });
  control.addEventListener("change", () => {
    clearFieldAutofillState(fieldId);
    updateFormProgress();
  });
});

if (reportDropzoneEl) {
  reportDropzoneEl.addEventListener("click", openFilePicker);
  reportDropzoneEl.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      openFilePicker();
    }
  });

  reportDropzoneEl.addEventListener("dragover", (event) => {
    event.preventDefault();
    reportDropzoneEl.classList.add("dragover");
  });

  reportDropzoneEl.addEventListener("dragleave", () => {
    reportDropzoneEl.classList.remove("dragover");
  });

  reportDropzoneEl.addEventListener("drop", (event) => {
    event.preventDefault();
    reportDropzoneEl.classList.remove("dragover");
    const [file] = event.dataTransfer.files || [];
    if (file) {
      setSelectedReportFile(file);
    }
  });
}

if (reportFileInputEl) {
  reportFileInputEl.addEventListener("change", () => {
    const [file] = reportFileInputEl.files || [];
    if (file) {
      setSelectedReportFile(file);
    }
  });
}

if (analyzeReportBtn) {
  analyzeReportBtn.addEventListener("click", analyzeSelectedReport);
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const payload = {
    age: Number(document.getElementById("age").value),
    gender: document.getElementById("gender").value,
    hypertension: document.getElementById("hypertension").value,
    heart_disease: document.getElementById("heart_disease").value,
    ever_married: document.getElementById("ever_married").value,
    work_type: document.getElementById("work_type").value,
    residence_type: document.getElementById("residence_type").value,
    avg_glucose_level: Number(document.getElementById("avg_glucose_level").value),
    bmi: document.getElementById("bmi").value ? Number(document.getElementById("bmi").value) : null,
    smoking_status: document.getElementById("smoking_status").value,
  };

  setSubmittingState(true);

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Prediction failed.");
    }

    renderResult(data);
  } catch (error) {
    alert(error.message);
  } finally {
    setSubmittingState(false);
  }
});

updateFormProgress();

// On index page load, check for report prefill data and apply it
try {
  const prefillRaw = localStorage.getItem("neuroscan_report_prefill");
  if (prefillRaw) {
    const prefill = JSON.parse(prefillRaw);
    if (prefill && typeof prefill === "object") {
      // Only apply when prediction form exists on the page
      const formEl = document.getElementById("prediction-form");
      if (formEl && typeof fillFieldsFromReport === "function") {
        fillFieldsFromReport(prefill.extracted_fields || {});
        if (prefill.prediction && typeof renderResult === "function") {
          renderResult(prefill.prediction);
        }
        // keep the prefill for user until they navigate away; remove to avoid reapplying next time
        localStorage.removeItem("neuroscan_report_prefill");
      }
    }
  }
} catch (e) {
  // ignore JSON/localStorage read errors
}
