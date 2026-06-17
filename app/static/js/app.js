/* =============================================================================
   app.js — Plant Disease Detection System Frontend Logic
   Handles: drag-drop upload, fetch prediction, render results
============================================================================= */

(function () {
  "use strict";

  // ── DOM refs ──────────────────────────────────────────────────────────────
  const dropZone    = document.getElementById("drop-zone");
  const fileInput   = document.getElementById("file-input");
  const previewImg  = document.getElementById("preview-img");
  const dropHint    = document.getElementById("drop-hint");
  const fileName    = document.getElementById("file-name");
  const analyseBtn  = document.getElementById("analyse-btn");
  const btnText     = document.getElementById("btn-text");
  const btnSpinner  = document.getElementById("btn-spinner");
  const errorBox    = document.getElementById("error-box");
  const resultsCard = document.getElementById("results-card");

  // Results fields
  const resultThumb    = document.getElementById("result-thumb");
  const resultTopName  = document.getElementById("result-top-name");
  const resultTopConf  = document.getElementById("result-top-conf");
  const resultSeverity = document.getElementById("result-severity");
  const resultWarning  = document.getElementById("result-warning");
  const predBars       = document.getElementById("pred-bars");
  const infoDesc       = document.getElementById("info-desc");
  const infoSymptoms   = document.getElementById("info-symptoms");
  const infoTreatment  = document.getElementById("info-treatment");

  let selectedFile = null;

  // ── File selection ────────────────────────────────────────────────────────

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

  dropZone.addEventListener("mouseenter", () => {
    dropZone.classList.add("hover");
  });

  dropZone.addEventListener("mouseleave", () => {
    dropZone.classList.remove("hover");
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
    previewImg.classList.add("preview-enter");
    dropHint.style.display   = "none";
    fileName.textContent     = `📎 ${file.name}`;

    analyseBtn.disabled = false;
    hideError();
    resultsCard.style.display = "none";
  }

  // ── Analyse ───────────────────────────────────────────────────────────────

  analyseBtn.addEventListener("click", async () => {
    if (!selectedFile) return;

    setLoading(true);
    hideError();
    resultsCard.style.display = "none";

    const form = new FormData();
    form.append("file", selectedFile);

    try {
      const res  = await fetch("/predict", { method: "POST", body: form });
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

  // ── Render results ────────────────────────────────────────────────────────

  function renderResults(data) {
    const top  = data.predictions[0];
    const info = data.disease_info || {};

    // Thumb + headline
    resultThumb.src        = data.image_data;
    resultTopName.textContent = top.display_name;
    resultTopConf.textContent = `Confidence: ${top.confidence}%`;

    const isUncertain = top.class_name === "uncertain_prediction" || top.confidence < 70;
    if (isUncertain) {
      resultWarning.textContent = `⚠️ Uncertain Prediction — Confidence: ${top.confidence}%\nThis image may not belong to any supported disease category.`;
      resultWarning.style.display = "block";
      resultSeverity.textContent = "Severity: Unknown";
      resultSeverity.style.color = "#b17704";
    } else {
      resultWarning.style.display = "none";
      const sev = info.severity || "";
      resultSeverity.textContent = sev ? `Severity: ${sev}` : "";
      resultSeverity.style.color =
        sev.toLowerCase().includes("high")      ? "#e63946" :
        sev.toLowerCase().includes("moderate")  ? "#e08a46" :
        sev.toLowerCase().includes("unknown")   ? "#b17704" :
        sev.toLowerCase().includes("none")      ? "#2d6a4f" : "#555";
    }

    // Prediction bars
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
          <div class="pred-fill ${i === 0 ? "top" : ""}"
               style="width:0%"
               data-w="${p.confidence}"></div>
        </div>`;
      predBars.appendChild(item);
    });

    // Animate bars after paint
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        document.querySelectorAll(".pred-fill").forEach((bar) => {
          bar.style.width = bar.dataset.w + "%";
        });
      });
    });

    // Top-line results
    document.getElementById("result-info-class").textContent      = top.display_name;
    document.getElementById("result-info-confidence").textContent = `${top.confidence}%`;
    document.getElementById("result-info-severity").textContent   = resultSeverity.textContent.replace("Severity: ", "") || "Unknown";

    // Disease info
    infoDesc.textContent      = info.description || "—";
    infoSymptoms.textContent  = info.symptoms    || "—";
    infoTreatment.textContent = info.treatment   || "Consult an agricultural expert.";

    resultsCard.style.display = "block";
  }

  // ── Helpers ───────────────────────────────────────────────────────────────

  function setLoading(on) {
    analyseBtn.disabled    = on;
    btnText.textContent    = on ? "Analyzing image..." : "Analyse Leaf";
    btnSpinner.style.display = on ? "inline-block" : "none";
  }

  function showError(msg) {
    errorBox.textContent   = "⚠️  " + msg;
    errorBox.style.display = "block";
  }

  function hideError() {
    errorBox.style.display = "none";
  }

})();
