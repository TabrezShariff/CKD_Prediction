const form = document.getElementById("ckd-form");
const steps = document.querySelectorAll(".form-group");
const prevBtn = document.getElementById("prev-btn");
const nextBtn = document.getElementById("next-btn");
const predictBtn = document.getElementById("predict-btn");
const stepCounter = document.getElementById("step-counter");
const progressFill = document.getElementById("progress-fill");

let currentStep = 0;

function showStep(step) {
  steps.forEach((el, idx) => {
    el.style.display = idx === step ? "block" : "none";
  });

  stepCounter.textContent = `Step ${step + 1} of ${steps.length}`;
  progressFill.style.width = `${((step + 1) / steps.length) * 100}%`;

  prevBtn.style.display = step === 0 ? "none" : "inline-block";
  nextBtn.style.display = step === steps.length - 1 ? "none" : "inline-block";
  predictBtn.style.display =
    step === steps.length - 1 ? "inline-block" : "none";
}

nextBtn.addEventListener("click", () => {
  if (currentStep < steps.length - 1) currentStep++;
  showStep(currentStep);
});

prevBtn.addEventListener("click", () => {
  if (currentStep > 0) currentStep--;
  showStep(currentStep);
});

predictBtn.addEventListener("click", async () => {
  const formData = new FormData(form);
  const response = await fetch("/predict", {
    method: "POST",
    body: formData,
  });
  const result = await response.json();
  const resultDiv = document.getElementById("result");
  const chart = document.getElementById("resultChart");

  if (result.error) {
    resultDiv.innerHTML = `<p style="color: red;">❌ ${result.error}</p>`;
  } else {
    resultDiv.innerHTML = `
            <h3 style="color: ${result.risk_color};">${result.risk_level}</h3>
            <p><strong>Probability:</strong> ${result.probability}</p>
            <p>${result.interpretation}</p>
            <p><strong>Confidence:</strong> ${result.confidence}</p>
            ${
              result.model_info
                ? `<div class="model-info"><strong>Model Accuracy:</strong> ${result.model_info.accuracy}<br><strong>Reliability:</strong> ${result.model_info.reliability}</div>`
                : ""
            }
            ${
              result.warnings
                ? `<div class="warning-message">⚠️ ${result.warnings.join(
                    "<br>"
                  )}</div>`
                : ""
            }
          `;

    if (window.resultChartInstance) {
      window.resultChartInstance.destroy();
    }

    window.resultChartInstance = new Chart(chart, {
      type: "doughnut",
      data: {
        labels: ["Risk", "Remaining"],
        datasets: [
          {
            data: [
              result.probability_decimal * 100,
              100 - result.probability_decimal * 100,
            ],
            backgroundColor: [result.risk_color, "#ecf0f1"],
          },
        ],
      },
      options: {
        responsive: true,
        cutout: "70%",
        plugins: {
          legend: { display: false },
        },
      },
    });
  }
});

// Initialize the first step
showStep(currentStep);
