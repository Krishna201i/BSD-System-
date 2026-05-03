const chartCanvas = document.getElementById("importance-chart");
const metricsGrid = document.getElementById("metrics-grid");

async function loadDashboard() {
  const response = await fetch("/feature-importance");
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || "Failed to load dashboard data.");
  }

  const labels = data.feature_importance.map((item) => item.feature_label);
  const values = data.feature_importance.map((item) => item.importance);

  // eslint-disable-next-line no-undef
  new Chart(chartCanvas, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "Feature Importance",
          data: values,
          backgroundColor: "rgba(14, 116, 144, 0.8)",
          borderRadius: 8,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      indexAxis: "y",
      plugins: {
        legend: { display: false },
      },
      scales: {
        x: {
          title: { display: true, text: "Importance Score" },
        },
      },
    },
  });

  metricsGrid.innerHTML = "";
  Object.entries(data.model_metrics).forEach(([key, value]) => {
    const card = document.createElement("div");
    card.className = "metric-item";

    const title = document.createElement("span");
    title.textContent = key.replaceAll("_", " ").toUpperCase();

    const metric = document.createElement("strong");
    metric.textContent = Number(value).toFixed(4);

    card.appendChild(title);
    card.appendChild(metric);
    metricsGrid.appendChild(card);
  });
}

loadDashboard().catch((error) => {
  alert(error.message);
});

