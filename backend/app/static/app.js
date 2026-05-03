const form = document.getElementById("prediction-form");
const loading = document.getElementById("loading");
const resultCard = document.getElementById("result-card");
const riskPercentageEl = document.getElementById("risk-percentage");
const riskBadgeEl = document.getElementById("risk-badge");
const factorListEl = document.getElementById("factor-list");

function showLoading(isLoading) {
  loading.classList.toggle("hidden", !isLoading);
}

function formatContribution(value) {
  return `${value > 0 ? "+" : ""}${value.toFixed(4)}`;
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

  showLoading(true);
  resultCard.classList.add("hidden");

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

    riskPercentageEl.textContent = `${data.risk_percentage.toFixed(2)}%`;
    riskBadgeEl.textContent = data.risk_label;
    riskBadgeEl.className = `badge ${data.risk_label === "High Risk" ? "high" : "low"}`;

    factorListEl.innerHTML = "";
    for (const factor of data.key_contributing_factors) {
      const item = document.createElement("li");
      item.textContent = `${factor.feature}: ${factor.direction} (${formatContribution(
        factor.contribution
      )})`;
      factorListEl.appendChild(item);
    }

    resultCard.classList.remove("hidden");
  } catch (error) {
    alert(error.message);
  } finally {
    showLoading(false);
  }
});

