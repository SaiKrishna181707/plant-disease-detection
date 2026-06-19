/* =============================================================================
   app.js � Frontend logic for Plant Disease Detection
   Handles upload, prediction fetch, and results rendering
============================================================================= */
(function () {
  "use strict";

  const dropZone = document.getElementById("drop-zone");
  const fileInput = document.getElementById("file-input");
  const previewImg = document.getElementById("preview-img");
  const fileName = document.getElementById("file-name");
  const analyseBtn = document.getElementById("analyse-btn");
  const btnText = document.getElementById("btn-text");
  const btnSpinner = document.getElementById("btn-spinner");
  const errorBox = document.getElementById("error-box");
  const resultsCard = document.getElementById("results-card");
  const resultThumb = document.getElementById("result-thumb");
  const resultTopName = document.getElementById("result-top-name");
  const resultTopConf = document.getElementById("result-top-conf");
  const resultSeverity = document.getElementById("result-severity");
  const resultWarning = document.getElementById("result-warning");
  const predBars = document.getElementById("pred-bars");
  const infoDesc = document.getElementById("info-desc");
  const infoSymptoms = document.getElementById("info-symptoms");
  const infoTreatment = document.getElementById("info-treatment");
  const resultInfoClass = document.getElementById("result-info-class");
  const resultInfoConfidence = document.getElementById("result-info-confidence");
  const resultInfoSeverity = document.getElementById("result-info-severity");

  let selectedFile = null;

  dropZone.addEventListener("click", () => fileInput.click());

  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) handleFile(e.target.files[0]);
  });

  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  });

  dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("dragover");
  });

  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
  });

  function handleFile(file) {
    const allowed = ["image/jpeg", "image/png", "image/webp", "image/bmp"];
    if (!allowed.includes(file.type)) {
      showError("Unsupported file type. Please upload JPG, PNG, WEBP, or BMP.");
      return;
    }
    if (file.size > 16 * 1024 * 1024) {
      showError("File too large. Maximum size is 16 MB.");
      return;
    }

    selectedFile = file;
    const url = URL.createObjectURL(file);
    previewImg.src = url;
    previewImg.style.display = "block";
    fileName.textContent = `?? ${file.name}`;
    analyseBtn.disabled = false;
    hideError();
    resultsCard.style.display = "none";
  }

  analyseBtn.addEventListener("click", async () => {
    if (!selectedFile) return;

    setLoading(true);
    hideError();
    resultsCard.style.display = "none";

    const form = new FormData();
    form.append("file", selectedFile);

    try {
      const res = await fetch("/predict", { method: "POST", body: form });
      const data = await res.json();
      if (!res.ok || data.error) {
        showError(data.error || "Server returned an error.");
        return;
      }
      renderResults(data);
    } catch (err) {
      showError("Network error: " + err.message);
    } finally {
      setLoading(false);
    }
  });

  function renderResults(data) {
    const top = data.predictions[0];
    const info = data.disease_info || {};

    resultThumb.src = data.image_data;
    resultTopName.textContent = top.display_name;
    resultTopConf.textContent = `Confidence: ${top.confidence}%`;
    

    const isLowConfidence = top.class_name === "uncertain_prediction" || top.confidence < 70;
    if (isLowConfidence) {
      resultWarning.textContent = "? Low Confidence Prediction � This image may not belong to a supported crop class or the image quality may be insufficient.";
      resultWarning.style.display = "block";
      resultSeverity.textContent = "Low confidence";
      resultSeverity.style.background = "rgba(250, 204, 21, 0.18)";
      resultSeverity.style.color = "#facc15";
      
      infoDesc.textContent = info.description || "Prediction confidence is low. Retake the image for a more reliable result.";
      infoSymptoms.textContent = info.symptoms || "Prediction confidence is low. Retake the image for a more reliable result.";
      infoTreatment.textContent = "Treatment recommendations are withheld for low confidence predictions.";
    } else {
      resultWarning.style.display = "none";
      const severity = info.severity || "Unknown";
      resultSeverity.textContent = severity;
      resultSeverity.style.background = "rgba(46, 125, 50, 0.18)";
      resultSeverity.style.color = "var(--success)";
      
      infoDesc.textContent = info.description || "�";
      infoSymptoms.textContent = info.symptoms || "�";
      infoTreatment.textContent = info.treatment || "Consult an agricultural expert.";
    }

    predBars.innerHTML = "";
    data.predictions.forEach((p, i) => {
      const item = document.createElement("div");
      item.className = "pred-item";
      item.innerHTML = `
        <div class="pred-header">
          <span class="pred-name">${p.display_name}</span>
          <span class="pred-conf">${p.confidence}%</span>
        </div>
        <div class="pred-bg">
          <div class="pred-fill ${i === 0 ? "top" : ""}" style="width:0%" data-w="${p.confidence}"></div>
        </div>`;
      predBars.appendChild(item);
    });

    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        document.querySelectorAll(".pred-fill").forEach((bar) => {
          bar.style.width = bar.dataset.w + "%";
        });
      });
    });

    resultsCard.style.display = "block";
  }

  function setLoading(on) {
    analyseBtn.disabled = on;
    btnText.textContent = on ? "Analyzing image..." : "Analyze Leaf";
    btnSpinner.style.display = on ? "inline-block" : "none";
  }

  function showError(msg) {
    errorBox.textContent = "? " + msg;
    errorBox.style.display = "block";
  }

  function hideError() {
    errorBox.style.display = "none";
  }
})();
