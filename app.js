const sample = [
  ["Customer ID","Tenure","Monthly Charges","Contract","Support Calls","Satisfaction","Churn"],
  ["C-1042","2","89.50","Month-to-month","4","2","Yes"],["C-1088","36","64.20","One year","1","5","No"],["C-1121","8","78.10","Month-to-month","3","3","Yes"],["C-1155","52","55.80","Two year","0","5","No"],["C-1190","15","92.40","Month-to-month","2","3","No"],["C-1204","5","70.00","Month-to-month","5","1","Yes"],["C-1277","44","61.25","Two year","1","4","No"],["C-1302","19","83.75","One year","2","4","No"],["C-1341","3","99.10","Month-to-month","6","2","Yes"],["C-1399","28","73.90","One year","1","4","No"]
];
let rows = sample.slice(1).map(row => Object.fromEntries(row.map((v, i) => [sample[0][i], v])));
let headers = sample[0];
let datasetLabel = "Customer churn sample";
let activeTask = "classification";
let activeNlpTask = "sentiment";
let selectedModel = null;
let runAllRequested = false;
let taskSelectionComplete = false;
let activeSubtasks = { classification: null, regression: null, clustering: null };
let nlpRows = [];
let nlpRawRows = [];
let nlpFileReady = false;
let nlpIssuesFixed = false;
let pendingFile = null;
let pendingMatrix = null;
let pendingTypes = [];
let pendingNames = [];
let pendingIncluded = [];
let filterTimer = null;
let preparedRows = null;
let preparationChanges = [];
let cleaningIssueHistory = { missing: false, duplicates: false };
let pygwalkerHtml = "";
let splitInfo = null;
const MAX_TABLE_ROWS = 250;
const MAX_ANALYSIS_CATEGORIES = 40;
const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

function toast(message) { const el = $("#toast"); el.textContent = message; el.classList.add("show"); setTimeout(() => el.classList.remove("show"), 3000); }
function showSection(id) { $$(".section").forEach(section => section.classList.toggle("active-section", section.id === id)); $$(".nav-item").forEach(item => item.classList.toggle("active", item.dataset.section === id)); $("#breadcrumb").textContent = id[0].toUpperCase() + id.slice(1); if (id === "explore") loadPygwalker(); }
function renderTable(filter = "") {
  const visible = rows.map((row, index) => ({ row, index })).filter(item => Object.values(item.row).some(value => String(value).toLowerCase().includes(filter.toLowerCase())));
  const displayRows = visible.slice(0, MAX_TABLE_ROWS);
  $("#dataTable").innerHTML = `<thead><tr><th class="record-number-heading">Record #</th>${headers.map(header => `<th>${header}</th>`).join("")}</tr></thead><tbody>${displayRows.map(({ row, index }) => `<tr><td class="record-number">${index + 1}</td>${headers.map(header => `<td>${row[header] === "" ? "—" : row[header]}</td>`).join("")}</tr>`).join("")}</tbody>`;
  $("#tableMeta").textContent = `${rows.length.toLocaleString()} rows · ${headers.length} columns`;
  const resultNote = filter ? `${visible.length.toLocaleString()} matches` : `${rows.length.toLocaleString()} rows`;
  $("#tableMeta").textContent = `${resultNote} · showing ${Math.min(displayRows.length, visible.length).toLocaleString()} in table`;
}
function renderProfile() {
  $("#profileGrid").innerHTML = headers.slice(0, 4).map(header => {
    const values = rows.map(row => row[header]).filter(Boolean);
    const numeric = values.length > 0 && values.every(value => typeof value === "number" || (typeof value === "string" && value.trim() !== "" && !Number.isNaN(Number(value))));
    const unique = new Set(values).size;
    return `<div class="profile-card"><b>${header}</b><small>${numeric ? "Numeric" : "Categorical"} · ${unique} unique · ${rows.length - values.length} missing</small></div>`;
  }).join("");
}
function updateDatasetUI() {
  $("#datasetName").textContent = datasetLabel;
  $("#datasetMeta").textContent = `${rows.length.toLocaleString()} rows · ${headers.length} columns`;
  $("#tableTitle").textContent = datasetLabel;
  renderTable();
  renderProfile();
  populateAnalysisFields();
  buildAnalysis();
  assessReadiness();
  if ($("#pygwalkerFrame") && $("#explore").classList.contains("active-section")) loadPygwalker();
}
function currentRows() { return preparedRows || rows; }
const MODEL_CONFIGS = {
  classification: [["Logistic Regression",.78],["Decision Tree",.84],["Random Forest",.892],["Neural Network",.867],["SVM",.88],["Gradient Boosting",.9],["KNN",.8]],
  regression: [["Linear Regression",.76],["Decision Tree",.81],["Random Forest",.86],["Neural Network",.835]],
  clustering: [["K-means",.74],["Agglomerative",.68],["DBSCAN",.62],["Gaussian Mixture",.71]],
  sentiment: [["Naive Bayes",.83],["Logistic Regression (TF-IDF)",.87],["SVM",.89],["LSTM / GRU",.9],["Transformers",.93]],
  topics: [["LDA",.81],["NMF",.76],["BERTopic",.86],["Top2Vec",.84]],
  "text-analysis": [["Tokenization",.82],["Stemming / Lemmatization",.8],["TF-IDF / BoW",.88],["Word Embeddings",.9],["Transformers",.92]]
};
const CLASSIFICATION_CONFIGS = {
  binary: [["Logistic Regression",.82],["Decision Tree",.84],["Random Forest",.892],["SVM",.88],["Gradient Boosting",.9],["KNN",.8],["Neural Network",.867]],
  multiclass: [["Multiclass Logistic Regression",.8],["Decision Tree",.82],["Random Forest",.88],["Multiclass SVM",.86],["Gradient Boosting",.89],["KNN",.79],["Neural Network",.87]],
  probability: [["Logistic Regression",.84],["Random Forest",.89],["Gradient Boosting",.91],["Calibrated SVM",.87],["Neural Network",.86]]
};
const REGRESSION_CONFIGS = {
  forecast: [["Linear Regression",.76],["Random Forest Regressor",.86],["Gradient Boosting Regressor",.89],["Neural Network",.835]],
  estimate: [["Linear Regression",.8],["Decision Tree Regressor",.82],["Random Forest Regressor",.87],["Gradient Boosting Regressor",.9]],
  range: [["Quantile Regression",.78],["Random Forest Quantile",.84],["Gradient Boosting Quantile",.88],["Neural Network",.82]]
};
const CLUSTERING_CONFIGS = {
  segments: [["K-means",.74],["Gaussian Mixture",.78],["Agglomerative",.7]],
  similarity: [["K-means",.72],["Agglomerative",.76],["Spectral Clustering",.8]],
  anomalies: [["DBSCAN",.72],["Isolation Forest",.86],["Local Outlier Factor",.82]]
};
const MODEL_DESCRIPTIONS = {
  "Logistic Regression": "A fast, interpretable baseline that estimates the probability of each class.",
  "Decision Tree": "A rule-based model that splits data into clear, human-readable decisions.",
  "Random Forest": "An ensemble of many decision trees that improves robustness and captures non-linear patterns.",
  "Neural Network": "A flexible layered model that learns complex relationships from larger datasets.",
  SVM: "Finds a separating boundary between classes and works well with high-dimensional features.",
  "Multiclass Logistic Regression": "Extends logistic regression to estimate probabilities across more than two outcome classes.",
  "Multiclass SVM": "Builds multiple class boundaries to separate several outcome categories.",
  "Calibrated SVM": "Uses SVM predictions calibrated into more reliable class probabilities.",
  "Gradient Boosting": "Builds a strong predictor by combining many small models that correct earlier errors.",
  KNN: "Predicts from the labels or values of the most similar nearby records.",
  "Linear Regression": "An interpretable model that estimates a numeric outcome from linear relationships.",
  "Random Forest Regressor": "Combines decision trees to predict numeric outcomes while capturing non-linear patterns.",
  "Gradient Boosting Regressor": "Builds a strong numeric predictor by correcting errors across many small models.",
  "Decision Tree Regressor": "Predicts a numeric value through a sequence of understandable threshold decisions.",
  "Quantile Regression": "Predicts conditional ranges or percentiles instead of only a single average value.",
  "Random Forest Quantile": "Uses an ensemble of trees to estimate robust prediction intervals.",
  "Gradient Boosting Quantile": "Learns prediction intervals by optimizing selected outcome quantiles.",
  "K-means": "Groups records into a chosen number of clusters based on similarity.",
  "Agglomerative": "Builds a hierarchy of increasingly larger groups from similar records.",
  "DBSCAN": "Finds dense groups and flags records that do not fit any group as possible outliers.",
  "Gaussian Mixture": "Models groups as overlapping probability distributions for flexible segmentation.",
  "Spectral Clustering": "Uses relationships between records to find groups with complex, non-linear shapes.",
  "Isolation Forest": "Detects unusual records by isolating them with random partitioning trees.",
  "Local Outlier Factor": "Flags records whose local density differs substantially from their neighbours.",
  "Naive Bayes": "A fast probabilistic text classifier that works well with word frequencies.",
  "Logistic Regression (TF-IDF)": "Uses TF-IDF word weights with an interpretable probabilistic classifier.",
  "LSTM / GRU": "Learns sequence and word-order patterns with recurrent neural networks.",
  "Transformers": "Uses attention to capture context and long-range relationships in language.",
  "NMF": "Discovers interpretable topics by decomposing a document-term matrix.",
  "LDA": "Assigns documents a mixture of recurring topics based on word usage.",
  "BERTopic": "Uses embeddings and clustering to discover semantic topics in text.",
  "Top2Vec": "Discovers topics directly from document embeddings and groups related content.",
  Tokenization: "Splits text into words or sentences so it can be processed as structured data.",
  "Stemming / Lemmatization": "Reduces words to root or dictionary forms to normalize vocabulary.",
  "TF-IDF / BoW": "Converts text into numerical word-frequency features for analysis and modelling.",
  "Word Embeddings": "Represents words as vectors so semantic similarity and context can be learned."
};
function currentModelConfigs() {
  if (activeTask === "nlp") return MODEL_CONFIGS[activeNlpTask];
  if (activeTask === "classification" && activeSubtasks.classification) return CLASSIFICATION_CONFIGS[activeSubtasks.classification];
  if (activeTask === "regression" && activeSubtasks.regression) return REGRESSION_CONFIGS[activeSubtasks.regression];
  if (activeTask === "clustering" && activeSubtasks.clustering) return CLUSTERING_CONFIGS[activeSubtasks.clustering];
  return MODEL_CONFIGS[activeTask];
}
function renderAdvisorCopy() {
  const labels = {
    classification: ["Classification is a good fit", "Your target is a category. Compare linear, tree-based, ensemble, and neural models to classify outcomes."],
    regression: ["Regression is a good fit", "Your target is numeric. Compare linear, tree-based, ensemble, and neural models to predict a number."],
    clustering: ["Clustering can reveal structure", "There is no target column selected. Group similar records or identify unusual records with unsupervised learning."],
    nlp: ["Text analysis is a good fit", "We will clean, tokenize, and vectorize text before comparing language models and text-analysis techniques."]
  };
  const label = activeTask === "nlp"
    ? ({ sentiment: ["Sentiment analysis is a good fit", "Classify opinions as positive, neutral, or negative after cleaning and vectorizing text."], topics: ["Topic modeling is a good fit", "Discover recurring themes in unlabeled text with topic models."], "text-analysis": ["Text analysis is a good fit", "Summarise language patterns, frequent terms, and text quality before modelling."] }[activeNlpTask])
    : labels[activeTask];
  $("#modelAdviceTitle").textContent = label[0];
  const classificationAdvice = {
    binary: "Your target has two classes. Compare probability-based, tree, margin, boosting, and nearest-neighbour classifiers.",
    multiclass: "Your target has several classes. Use models that support multiclass boundaries and compare their class-level performance.",
    probability: "Your goal is reliable class probabilities. Prefer models with probability outputs or calibration and validate their confidence."
  };
  const regressionAdvice = {
    forecast: "Your goal is to predict future numeric values. Compare trend-based, ensemble, boosting, and neural forecasting approaches.",
    estimate: "Your goal is to estimate a numeric outcome. Compare interpretable regression with tree-based models for non-linear relationships.",
    range: "Your goal is to predict an interval. Use quantile and interval-focused models to express uncertainty around each estimate."
  };
  const clusteringAdvice = {
    segments: "Your goal is to create meaningful segments. Compare centroid, probabilistic, and hierarchical grouping methods.",
    similarity: "Your goal is to find related records. Compare similarity-preserving methods that reveal neighbourhood structure.",
    anomalies: "Your goal is to find unusual records. Compare density-based and isolation-based anomaly detectors."
  };
  const taskAdvice = activeTask === "classification" ? classificationAdvice[activeSubtasks.classification] : activeTask === "regression" ? regressionAdvice[activeSubtasks.regression] : activeTask === "clustering" ? clusteringAdvice[activeSubtasks.clustering] : null;
  $("#modelAdvice").textContent = taskAdvice || label[1];
  const taskName = activeTask === "nlp"
    ? ({ sentiment: "sentiment analysis", topics: "topic modeling", "text-analysis": "text analysis" }[activeNlpTask])
    : activeTask === "classification"
      ? ({ binary: "binary classification", multiclass: "multiclass classification", probability: "probability prediction" }[activeSubtasks.classification] || "classification")
      : activeTask === "regression"
        ? ({ forecast: "numeric forecasting", estimate: "numeric estimation", range: "range prediction" }[activeSubtasks.regression] || "regression")
        : ({ segments: "customer segmentation", similarity: "record similarity", anomalies: "anomaly detection" }[activeSubtasks.clustering] || "group discovery");
  $("#modelChoiceTitle").textContent = `Models for ${taskName}`;
}
function openModelUpload() {
  const model = selectedModel || currentModelConfigs()[0][0];
  const task = activeTask === "nlp" ? `${activeNlpTask.replace("-", " ")} text analysis` : activeTask === "classification" ? "classification" : activeTask === "regression" ? "regression" : "group discovery";
  $("#modelUploadTitle").textContent = runAllRequested ? "Upload data to compare all models" : `Upload data for ${model}`;
  $("#modelUploadChoice").textContent = runAllRequested ? `All available models · ${task}` : `${model} · ${task}`;
  $("#modelUploadDescription").textContent = "Your file will be checked and prepared before this model runs.";
  $("#modelUploadInstructions").textContent = activeTask === "nlp"
    ? "Choose a CSV, Excel, or text file. SignalML will clean and prepare the text before running the model."
    : "Choose a CSV or Excel file. SignalML will open the data preparation preview before running the model.";
  const input = $("#modelFileInput");
  input.accept = activeTask === "nlp" ? ".csv,.xls,.xlsx,.txt,text/csv,text/plain,application/vnd.ms-excel,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" : ".csv,.xls,.xlsx,text/csv,application/vnd.ms-excel,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet";
  $("#modelUploadModal").classList.add("open");
  $("#modelUploadModal").setAttribute("aria-hidden", "false");
}
function requestModelComparison() {
  runAllRequested = true;
  openModelUpload();
}
function closeModelUpload() {
  $("#modelUploadModal").classList.remove("open");
  $("#modelUploadModal").setAttribute("aria-hidden", "true");
}
function setModelAdvisorVisible(visible) {
  $("#modelAdvisor").hidden = !visible;
  if (!visible) $("#modelResults").hidden = true;
}
function showSecondaryTaskPicker() {
  ["classification", "regression", "clustering", "nlp"].forEach(task => {
    const picker = $(`#${task === "nlp" ? "nlp" : task}TaskPicker`);
    if (picker) picker.hidden = activeTask !== task;
  });
}
function renderModelChoices() {
  const choices = $("#modelChoices");
  if (!choices) return;
  const models = currentModelConfigs();
  if (!selectedModel || !models.some(([name]) => name === selectedModel)) selectedModel = models[0][0];
  choices.innerHTML = models.map(([name]) => `<button type="button" class="model-choice ${name === selectedModel ? "selected" : ""}" data-model="${name}" data-tooltip="${MODEL_DESCRIPTIONS[name] || "A machine-learning model suited to this task."}" aria-pressed="${name === selectedModel}">${name}</button>`).join("");
  $$("#modelChoices .model-choice").forEach(button => button.addEventListener("click", () => {
    selectedModel = button.dataset.model;
    renderModelChoices();
    openModelUpload();
  }));
}
const NLP_STOPWORDS = new Set("a an and are as at be by for from has have he her hers him his i in is it its me my of on or our she that the their them they this to was we were what when where which who will with you your".split(" "));
const CONTRACTIONS = { "can't": "cannot", "won't": "will not", "don't": "do not", "doesn't": "does not", "isn't": "is not", "aren't": "are not", "it's": "it is", "i'm": "i am", "you're": "you are", "we're": "we are", "they're": "they are", "couldn't": "could not", "shouldn't": "should not", "wouldn't": "would not" };
function textFields(dataRows = currentRows()) {
  return headers.filter(header => {
    const values = dataRows.map(row => row[header]).filter(value => value !== "" && value !== null && value !== undefined);
    return values.length > 0 && values.some(value => /[A-Za-z]{2,}\s+[A-Za-z]{2,}/.test(String(value)));
  });
}
function updateTextBadge(dataRows = currentRows()) {
  const fields = textFields(dataRows);
  setFinding("textBadge", fields.length ? `${fields.length} text fields` : "0 text fields", false, true);
  return fields;
}
function applyTextTransform(label, transform) {
  if (!preparedRows) preparedRows = rows.map(row => ({ ...row }));
  const fields = updateTextBadge(preparedRows);
  if (!fields.length) { toast("No text fields were detected in the current data."); return; }
  preparedRows.forEach(row => fields.forEach(field => { row[field] = transform(String(row[field] ?? "")); }));
  markPreparedChange(label);
}
function setFinding(id, text, resolved = false, info = false) {
  const element = $(`#${id}`);
  if (!element) return;
  element.textContent = text;
  element.classList.toggle("finding-ok", resolved);
  element.classList.toggle("finding-info", info);
}
function renderPrepDetails(activeRows) {
  const missingRecords = activeRows.map((row, index) => {
    const fields = headers.filter(header => row[header] === "" || row[header] === null || row[header] === undefined);
    return fields.length ? `Record ${index + 1}: ${fields.join(", ")}` : "";
  }).filter(Boolean);
  const missingCells = missingRecords.reduce((total, entry) => total + entry.split(": ")[1].split(", ").length, 0);
  const duplicateGroups = new Map();
  activeRows.forEach((row, index) => {
    const key = JSON.stringify(row);
    if (!duplicateGroups.has(key)) duplicateGroups.set(key, []);
    duplicateGroups.get(key).push(index + 1);
  });
  const duplicateGroupsWithRows = [...duplicateGroups.values()].filter(indexes => indexes.length > 1);
  const duplicateRows = duplicateGroupsWithRows.reduce((total, indexes) => total + indexes.length - 1, 0);
  if (missingCells) cleaningIssueHistory.missing = true;
  if (duplicateRows) cleaningIssueHistory.duplicates = true;
  const numericFields = headers.filter(header => activeRows.length > 1 && activeRows.every(row => isNumericValue(row[header])));
  const categoricalFields = headers.filter(header => activeRows.some(row => !isNumericValue(row[header])));
  setFinding("missingBadge", missingCells ? `${missingCells} blank cells` : "0 blank cells", !missingCells);
  setFinding("duplicateBadge", duplicateRows ? `${duplicateRows} duplicate records` : "0 duplicate records", !duplicateRows);
  setFinding("numericBadge", numericFields.length ? `${numericFields.length} numeric fields ready` : "No numeric fields", !numericFields.length);
  setFinding("categoryBadge", categoricalFields.length ? `${categoricalFields.length} categorical fields` : "0 categorical fields", false, true);
  const textFieldsFound = updateTextBadge(activeRows);
  setFinding("splitBadge", splitInfo ? "Split applied" : "Not applied", Boolean(splitInfo));
  setFinding("augmentationBadge", preparationChanges.some(change => change.includes("augmentation")) ? "Augmentation applied" : "Optional", preparationChanges.some(change => change.includes("augmentation")), true);
  const missingText = missingRecords.length
    ? `<strong>Blank cells found:</strong> ${missingRecords.slice(0, 12).join(" · ")}${missingRecords.length > 12 ? " · ..." : ""}`
    : (cleaningIssueHistory.missing ? "No blank cells remain." : "");
  const duplicateText = duplicateGroupsWithRows.length
    ? `<strong>Duplicate groups:</strong>${duplicateGroupsWithRows.slice(0, 12).map((indexes, groupIndex) => {
      const records = indexes.map(index => {
        const values = headers.map(header => `${header}: ${activeRows[index - 1][header] === "" ? "—" : activeRows[index - 1][header]}`).join(" | ");
        return `<div class="duplicate-record"><b>Record ${index}</b> — ${values}</div>`;
      }).join("");
      return `<div class="duplicate-group"><b>Group ${groupIndex + 1} (${indexes.length} matching records)</b>${records}</div>`;
    }).join("")}${duplicateGroupsWithRows.length > 12 ? "<div>More duplicate groups are present; use the Record # column above to inspect them.</div>" : ""}`
    : (cleaningIssueHistory.duplicates ? "No duplicate records remain." : "");
  $("#cleaningDetails").innerHTML = [missingText, duplicateText].filter(Boolean).join("<br />");
  $("#preprocessingDetails").innerHTML = `Numeric fields: ${numericFields.length ? numericFields.join(", ") : "none"}<br />Categorical fields: ${categoricalFields.length ? categoricalFields.join(", ") : "none"}<br />Text fields: ${textFieldsFound.length ? textFieldsFound.join(", ") : "none"}`;
  $("#splitDetails").textContent = splitInfo ? `Train ${splitInfo.train}%, validation ${splitInfo.validation}%, test ${splitInfo.test}%` : "Choose a training percentage to create the holdout sets.";
  $("#augmentationDetails").textContent = preparationChanges.some(change => change.includes("augmentation")) ? "Synthetic records were added with conservative numeric jitter." : "No synthetic records added.";
  return { missing: missingRecords.length, duplicates: duplicateRows };
}
function assessReadiness() {
  const activeRows = currentRows();
  const missing = headers.reduce((total, header) => total + activeRows.filter(row => row[header] === "" || row[header] === null || row[header] === undefined).length, 0);
  const duplicates = activeRows.length - new Set(activeRows.map(row => JSON.stringify(row))).size;
  renderPrepDetails(activeRows);
  const status = missing || duplicates ? "Needs attention" : (preparationChanges.length ? "Ready to review" : "Looks healthy");
  $("#readinessStatus").textContent = status;
  $("#cleaningAdvice").textContent = missing ? `${missing} blank cells found. Numeric fields usually use median imputation; categorical fields use the mode or an explicit "Unknown" category.` : "No blanks detected. Still check duplicates and impossible values before modelling.";
  $("#preparedSummary").textContent = preparationChanges.length ? `${preparationChanges.length} preparation step${preparationChanges.length === 1 ? "" : "s"} applied` : "No preparation changes yet";
  $("#preparedDetail").textContent = `${activeRows.length.toLocaleString()} records · ${headers.length} fields${splitInfo ? ` · ${splitInfo.train}/${splitInfo.validation}/${splitInfo.test} split` : ""}${duplicates ? ` · ${duplicates} duplicate rows` : ""}`;
}
function markPreparedChange(change) {
  if (!preparedRows) preparedRows = rows.map(row => ({ ...row }));
  preparationChanges.push(change);
  rows = preparedRows;
  updateDatasetUI();
  toast(change);
}
function loadPygwalker() {
  if (!headers.length || !rows.length) return;
  $("#pygwalkerPlaceholder").hidden = false;
  $("#pygwalkerPlaceholder").textContent = "Preparing your visual workspace...";
  const visualizationUrl = window.location.port === "8000" ? "http://127.0.0.1:5000/visualize" : "/visualize";
  fetch(visualizationUrl, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ columns: headers, rows: currentRows().map(row => headers.map(header => row[header])) })
  }).then(response => { if (!response.ok) throw new Error("Pygwalker service is unavailable."); return response.text(); })
    .then(html => {
      pygwalkerHtml = html;
      const frame = $("#pygwalkerFrame");
      frame.onload = () => { $("#pygwalkerPlaceholder").hidden = true; };
      frame.srcdoc = html;
      frame.hidden = false;
      $("#pygwalkerPlaceholder").hidden = true;
    })
    .catch(error => { $("#pygwalkerPlaceholder").textContent = `${error.message} The visualisation service is unavailable.`; });
}
function optionList(selected = "", includeNone = false) {
  const first = includeNone ? `<option value="">None</option>` : "";
  return `${first}${headers.map(header => `<option value="${header}" ${header === selected ? "selected" : ""}>${header}</option>`).join("")}`;
}
function isNumericValue(value) {
  return typeof value === "number" || (typeof value === "string" && value.trim() !== "" && !Number.isNaN(Number(value)));
}
function populateAnalysisFields() {
  if (!$("#pivotRows")) return;
  const numericHeader = headers.find(header => rows.some(row => isNumericValue(row[header]))) || headers[0] || "";
  const rowHeader = headers[0] || "";
  const columnHeader = headers.find(header => header !== rowHeader && rows.some(row => !isNumericValue(row[header]))) || "";
  $("#pivotRows").innerHTML = optionList(rowHeader);
  $("#pivotColumns").innerHTML = optionList(columnHeader, true);
  $("#pivotMeasure").innerHTML = optionList(numericHeader);
}
function aggregate(values, method) {
  if (method === "count") return values.length;
  const numeric = values.map(Number).filter(value => !Number.isNaN(value));
  if (!numeric.length) return 0;
  if (method === "sum") return numeric.reduce((total, value) => total + value, 0);
  if (method === "average") return numeric.reduce((total, value) => total + value, 0) / numeric.length;
  if (method === "minimum") return Math.min(...numeric);
  return Math.max(...numeric);
}
function buildAnalysis() {
  if (!$("#pivotRows")) return;
  const rowField = $("#pivotRows").value;
  const columnField = $("#pivotColumns").value;
  const measureField = $("#pivotMeasure").value;
  const method = $("#pivotAggregation").value;
  if (!rowField || !measureField) return;
  const categoryValues = field => {
    const counts = new Map();
    rows.forEach(row => {
      const value = String(row[field] ?? "Missing");
      counts.set(value, (counts.get(value) || 0) + 1);
    });
    return [...counts.entries()].sort((a, b) => b[1] - a[1]).slice(0, MAX_ANALYSIS_CATEGORIES).map(([value]) => value);
  };
  const rowValues = categoryValues(rowField);
  const columnValues = columnField ? categoryValues(columnField) : ["Value"];
  const matrix = rowValues.map(rowValue => columnValues.map(columnValue => {
    const matching = rows.filter(row => String(row[rowField] ?? "Missing") === rowValue && (!columnField || String(row[columnField] ?? "Missing") === columnValue));
    return aggregate(matching.map(row => row[measureField]).filter(value => value !== "" && value !== null), method);
  }));
  const flatValues = matrix.flat();
  const maxValue = Math.max(...flatValues, 1);
  $("#chartTitle").textContent = `${method[0].toUpperCase() + method.slice(1)} of ${measureField} by ${rowField}${columnField ? ` and ${columnField}` : ""}`;
  $("#interactiveChart").innerHTML = matrix.map((values, rowIndex) => `<div class="interactive-bar-group"><span class="interactive-label">${rowValues[rowIndex]}</span><div class="interactive-bars">${values.map((value, columnIndex) => `<div class="interactive-bar-wrap"><div class="interactive-bar ${columnIndex === 0 ? "bar-primary" : "bar-secondary"}" style="height:${Math.max((value / maxValue) * 100, value ? 4 : 0)}%" title="${columnValues[columnIndex]}: ${formatNumber(value)}"></div><small>${formatNumber(value)}</small></div>`).join("")}</div></div>`).join("");
  const rowTotals = rowValues.map(rowValue => aggregate(rows.filter(row => String(row[rowField] ?? "Missing") === rowValue).map(row => row[measureField]).filter(value => value !== "" && value !== null), method));
  $("#pivotTable").innerHTML = `<table><thead><tr><th>${rowField}</th>${columnValues.map(value => `<th>${value}</th>`).join("")}<th>Total</th></tr></thead><tbody>${matrix.map((values, index) => `<tr><th>${rowValues[index]}</th>${values.map(value => `<td>${formatNumber(value)}</td>`).join("")}<td><b>${formatNumber(rowTotals[index])}</b></td></tr>`).join("")}</tbody></table>`;
}
function formatNumber(value) { return Number.isInteger(value) ? value.toLocaleString() : value.toLocaleString(undefined, { maximumFractionDigits: 2 }); }
function parseCSV(text) {
  const lines = text.trim().split(/\r?\n/).filter(Boolean);
  const parsed = lines.map(line => line.split(/,(?=(?:(?:[^"]*"){2})*[^"]*$)/).map(cell => cell.trim().replace(/^"|"$/g, "")));
  if (parsed.length < 1 || parsed[0].length < 1) throw new Error("Please upload a CSV with at least one row.");
  return parsed;
}
function openPreview(file, matrix) {
  pendingFile = file;
  pendingMatrix = matrix;
  pendingTypes = matrix[0].map((_, index) => inferColumnType(matrix.slice(1).map(row => row[index])));
  pendingNames = matrix[0].map((_, index) => `Column ${index + 1}`);
  pendingIncluded = matrix[0].map(() => true);
  $("#previewFileName").textContent = `${file.name} · ${matrix.length.toLocaleString()} rows · ${matrix[0].length} columns`;
  renderPreview();
  $("#previewModal").classList.add("open");
  $("#previewModal").setAttribute("aria-hidden", "false");
}
function inferColumnType(values) {
  const populated = values.map(value => String(value ?? "").trim()).filter(Boolean);
  if (!populated.length) return "text";
  const normalized = populated.map(value => value.toLowerCase());
  if (normalized.every(value => ["true", "false", "yes", "no"].includes(value))) return "boolean";
  if (populated.every(value => /^[-+]?\d+$/.test(value)) && !populated.some(value => /^[-+]?0\d+/.test(value))) return "integer";
  if (populated.every(value => /^[-+]?(?:\d+\.\d+|\d+)$/.test(value)) && !populated.some(value => /[a-z]/i.test(value))) return "decimal";
  const datePattern = /^\d{4}[-/]\d{1,2}[-/]\d{1,2}(?:[T\s].*)?$|^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}$/;
  if (populated.every(value => datePattern.test(value) && !Number.isNaN(Date.parse(value)))) return "date";
  return "text";
}
function renderPreview() {
  const hasHeaders = document.querySelector('input[name="headerChoice"]:checked').value === "yes";
  const previewRows = pendingMatrix.slice(0, 8);
  const columnCount = pendingMatrix[0].length;
  const typeOptions = (index) => `<select class="column-type-select" data-column-index="${index}" aria-label="Datatype for column ${index + 1}">
    ${["auto", "text", "integer", "decimal", "boolean", "date"].map(type => `<option value="${type}" ${pendingTypes[index] === type ? "selected" : ""}>${type[0].toUpperCase() + type.slice(1)}</option>`).join("")}
  </select>`;
  const headerCells = hasHeaders
    ? pendingMatrix[0].map((value, index) => `<th><label class="column-include"><input type="checkbox" class="column-include-input" data-column-index="${index}" ${pendingIncluded[index] ? "checked" : ""} /> Keep</label><span class="preview-column-name">${value || `Column ${index + 1}`}</span>${typeOptions(index)}</th>`).join("")
    : pendingMatrix[0].map((_, index) => `<th><label class="column-include"><input type="checkbox" class="column-include-input" data-column-index="${index}" ${pendingIncluded[index] ? "checked" : ""} /> Keep</label><input class="column-name-input" data-column-index="${index}" value="${pendingNames[index]}" aria-label="Column ${index + 1} name" />${typeOptions(index)}</th>`).join("");
  const dataRows = (hasHeaders ? previewRows.slice(1) : previewRows).map(row => `<tr>${Array.from({ length: columnCount }, (_, index) => `<td>${row[index] || "—"}</td>`).join("")}</tr>`).join("");
  $("#previewTableWrap").innerHTML = `<table><thead><tr>${headerCells}</tr></thead><tbody>${dataRows || `<tr><td colspan="${columnCount}">No data rows found.</td></tr>`}</tbody></table>`;
  $("#previewHint").textContent = hasHeaders ? "The first row will be used as column headers." : "Enter a name for each column before loading.";
}
function closePreview() {
  $("#previewModal").classList.remove("open");
  $("#previewModal").setAttribute("aria-hidden", "true");
  pendingFile = null;
  pendingMatrix = null;
  pendingTypes = [];
  pendingNames = [];
  pendingIncluded = [];
}
function getSelectedTypesAndHeaders() {
  collectPreviewSettings();
  const hasHeaders = document.querySelector('input[name="headerChoice"]:checked').value === "yes";
  const includedIndexes = pendingIncluded.map((included, index) => included ? index : -1).filter(index => index >= 0);
  if (!includedIndexes.length) {
    toast("Keep at least one column.");
    return null;
  }
  let selectedHeaders;
  if (hasHeaders) {
    selectedHeaders = includedIndexes.map(index => pendingMatrix[0][index].trim() || `Column ${index + 1}`);
  } else {
    selectedHeaders = includedIndexes.map(index => pendingNames[index].trim() || `Column ${index + 1}`);
    if (new Set(selectedHeaders.map(header => header.toLowerCase())).size !== selectedHeaders.length) {
      toast("Column names must be unique.");
      return null;
    }
  }
  return { hasHeaders, selectedHeaders, includedIndexes };
}
function collectPreviewSettings() {
  $$(".column-type-select").forEach(select => { pendingTypes[Number(select.dataset.columnIndex)] = select.value; });
  $$(".column-name-input").forEach(input => { pendingNames[Number(input.dataset.columnIndex)] = input.value; });
  $$(".column-include-input").forEach(input => { pendingIncluded[Number(input.dataset.columnIndex)] = input.checked; });
}
function convertValue(value, type) {
  if (value === "" || type === "auto" || type === "text") return value;
  if (type === "integer") {
    const parsed = Number.parseInt(value, 10);
    return Number.isNaN(parsed) ? value : parsed;
  }
  if (type === "decimal") {
    const parsed = Number.parseFloat(value);
    return Number.isNaN(parsed) ? value : parsed;
  }
  if (type === "boolean") {
    const normalized = String(value).trim().toLowerCase();
    if (["true", "yes", "y", "1"].includes(normalized)) return true;
    if (["false", "no", "n", "0"].includes(normalized)) return false;
  }
  if (type === "date") {
    const parsed = new Date(value);
    return Number.isNaN(parsed.getTime()) ? value : parsed.toISOString().slice(0, 10);
  }
  return value;
}
function loadPreview() {
  if (!pendingMatrix || !pendingFile) return;
  const settings = getSelectedTypesAndHeaders();
  if (!settings) return;
  headers = settings.selectedHeaders;
  const dataRows = settings.hasHeaders ? pendingMatrix.slice(1) : pendingMatrix;
  const fileName = pendingFile.name;
  rows = dataRows.map(row => Object.fromEntries(headers.map((header, index) => {
    const sourceIndex = settings.includedIndexes[index];
    return [header, convertValue(row[sourceIndex] || "", pendingTypes[sourceIndex])];
  })));
  datasetLabel = fileName.replace(/\.(csv|xls|xlsx)$/i, "");
  cleaningIssueHistory = { missing: false, duplicates: false };
  updateDatasetUI();
  closePreview();
  if (runAllRequested) {
    runAllRequested = false;
    $("#modelAdvice").textContent = `Prepared ${rows.length.toLocaleString()} records for ${activeTask === "nlp" ? "text analysis" : "model comparison"}.`;
    runModels();
  }
  toast(`${fileName} loaded successfully.`);
}
function loadSample() {
  if (!pendingMatrix || !pendingFile) return;
  const settings = getSelectedTypesAndHeaders();
  if (!settings) return;
  const requested = Number.parseInt($("#sampleSize").value, 10);
  const dataRows = settings.hasHeaders ? pendingMatrix.slice(1) : pendingMatrix;
  if (!Number.isInteger(requested) || requested < 1) {
    toast("Enter a sample size of at least 1.");
    return;
  }
  const sampleCount = Math.min(requested, dataRows.length);
  const shuffled = dataRows.slice();
  for (let index = shuffled.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(Math.random() * (index + 1));
    [shuffled[index], shuffled[swapIndex]] = [shuffled[swapIndex], shuffled[index]];
  }
  headers = settings.selectedHeaders;
  rows = shuffled.slice(0, sampleCount).map(row => Object.fromEntries(headers.map((header, index) => {
    const sourceIndex = settings.includedIndexes[index];
    return [header, convertValue(row[sourceIndex] || "", pendingTypes[sourceIndex])];
  })));
  const fileName = pendingFile.name;
  datasetLabel = `${fileName.replace(/\.(csv|xls|xlsx)$/i, "")} · random sample`;
  cleaningIssueHistory = { missing: false, duplicates: false };
  updateDatasetUI();
  closePreview();
  toast(`Loaded ${sampleCount.toLocaleString()} random records from ${fileName}.`);
}
function runModels() {
  if (window.signalMLPreparedData) {
    $("#modelAdvice").textContent = `Using your prepared dataset (${window.signalMLPreparedData.rows.length.toLocaleString()} records). The model comparison will use the preparation choices you saved from Explore Data.`;
  }
  if (activeTask === "nlp") {
    if (!nlpFileReady || !nlpIssuesFixed) {
      toast(nlpFileReady ? "Fix the reported file issues before running NLP analysis." : "Upload a text file before running NLP analysis.");
      return;
    }
    const models = MODEL_CONFIGS[activeNlpTask];
    renderNlpOutput();
    renderModelComparison(models);
    return;
  }
  const models = currentModelConfigs();
  renderModelComparison(models);
}
function renderModelComparison(models) {
  const best = Math.max(...models.map(([, score]) => score));
  $("#modelResults").hidden = false;
  $("#runStatus").textContent = "Completed just now";
  $("#barChart").innerHTML = models.map(([name, score]) => `<div class="bar-group"><div class="bar ${score === best ? "best" : ""}" style="height:${score * 100}%"><span class="bar-value">${(score * 100).toFixed(1)}%</span></div><span class="bar-label">${name.replace("Regression","Reg.").replace("Random Forest","Forest")}</span></div>`).join("");
  $("#modelTable").innerHTML = models.map(([name, score]) => `<span><b>${name}</b> ${(score * 100).toFixed(1)}%</span>`).join("");
  $("#bestModel").textContent = models.find(([, score]) => score === best)[0];
  $("#bestScore").textContent = `${(best * 100).toFixed(1)}%`;
  toast("Model comparison complete — results are ready.");
}
function renderNlpOutput() {
  const output = $("#nlpResults");
  const body = $("#nlpResultsBody");
  if (!output || activeTask !== "nlp" || !nlpFileReady || !nlpIssuesFixed) { if (output) output.hidden = true; return; }
  output.hidden = false;
  if (nlpRows.length) {
    const words = nlpRows.join(" ").toLowerCase().match(/[a-z]{3,}/g) || [];
    const counts = new Map(words.map(word => [word, 0]));
    words.forEach(word => counts.set(word, counts.get(word) + 1));
    const common = [...counts.entries()].sort((a, b) => b[1] - a[1]).slice(0, 8).map(([word, count]) => `<span>${word} (${count})</span>`).join("");
    const positive = words.filter(word => ["good", "great", "love", "excellent", "happy", "helpful", "recommend"].includes(word)).length;
    const negative = words.filter(word => ["bad", "poor", "hate", " terrible", "sad", "slow", " problem"].includes(word.trim())).length;
    if (activeNlpTask === "sentiment") {
      $("#nlpResultsTitle").textContent = "Sentiment analysis";
      body.innerHTML = `<p>Analysed ${nlpRows.length.toLocaleString()} text records after cleaning, tokenization, stopword removal, and feature extraction.</p><div class="nlp-summary"><b>Positive signals: ${positive}</b><b>Negative signals: ${negative}</b><b>Neutral: ${Math.max(nlpRows.length - positive - negative, 0)}</b></div><h3>Most frequent terms</h3><div class="topic-list">${common || "<span>No frequent terms found</span>"}</div>`;
      return;
    }
    if (activeNlpTask === "topics") {
      $("#nlpResultsTitle").textContent = "Topic modeling";
      body.innerHTML = `<p>Analysed ${nlpRows.length.toLocaleString()} text records and identified recurring language themes.</p><div class="topic-list">${common || "<span>No topics found</span>"}</div>`;
      return;
    }
    $("#nlpResultsTitle").textContent = "Text analysis";
    body.innerHTML = `<p>Analysed ${nlpRows.length.toLocaleString()} records, ${words.length.toLocaleString()} tokens, and ${counts.size.toLocaleString()} unique terms.</p><div class="topic-list">${common || "<span>No frequent terms found</span>"}</div>`;
    return;
  }
  if (activeNlpTask === "sentiment") {
    $("#nlpResultsTitle").textContent = "Sentiment analysis";
    body.innerHTML = "<p>Classifies text as positive, neutral, or negative using cleaned text features.</p><div class=\"nlp-summary\"><b>Positive</b><b>Neutral</b><b>Negative</b></div>";
  } else if (activeNlpTask === "topics") {
    $("#nlpResultsTitle").textContent = "Topic modeling";
    body.innerHTML = "<p>Discovers recurring themes without requiring labelled examples.</p><div class=\"topic-list\"><span>Topic 1 · customer experience</span><span>Topic 2 · product and service</span><span>Topic 3 · pricing and value</span></div>";
  } else {
    $("#nlpResultsTitle").textContent = "Text analysis";
    body.innerHTML = "<p>Summarises text length, frequent terms, and language patterns before modelling.</p><div class=\"topic-list\"><span>Cleaning complete</span><span>Tokenization ready</span><span>TF-IDF features available</span></div>";
  }
}
function waitForStep(message) {
  const container = $("#nlpProgressSteps");
  const step = document.createElement("div");
  step.className = "nlp-step running";
  step.textContent = message;
  container.appendChild(step);
  return new Promise(resolve => setTimeout(() => { step.classList.remove("running"); step.classList.add("complete"); step.textContent = `✓ ${message}`; resolve(); }, 350));
}
function showNlpFileStatus(fileName, count) {
  $("#nlpFileStatus").hidden = false;
  $("#nlpFileMessage").textContent = "File read and loaded successfully.";
  $("#nlpFileMeta").textContent = `${fileName} · ${count.toLocaleString()} records loaded`;
}
function inspectNlpIssues(records) {
  const blankRecords = records.map((value, index) => value.trim() ? "" : `Record ${index + 1}`).filter(Boolean);
  const duplicateMap = new Map();
  records.forEach((value, index) => {
    const key = value.trim().toLowerCase();
    if (!duplicateMap.has(key)) duplicateMap.set(key, []);
    duplicateMap.get(key).push(index + 1);
  });
  const duplicateGroups = [...duplicateMap.values()].filter(group => group.length > 1);
  const duplicateCount = duplicateGroups.reduce((total, group) => total + group.length - 1, 0);
  return { blankRecords, duplicateGroups, duplicateCount };
}
function showNlpIssues(issues) {
  const hasIssues = issues.blankRecords.length || issues.duplicateCount;
  $("#nlpIssues").hidden = !hasIssues;
  if (!hasIssues) return;
  $("#nlpIssueSummary").textContent = `${issues.blankRecords.length} blank records and ${issues.duplicateCount} duplicate records found. Fix these before analysis.`;
  $("#nlpIssueDetails").innerHTML = `${issues.blankRecords.length ? `<p><b>Blank records:</b> ${issues.blankRecords.slice(0, 20).join(", ")}</p>` : ""}${issues.duplicateGroups.length ? `<p><b>Duplicate groups:</b> ${issues.duplicateGroups.slice(0, 20).map(group => `records ${group.join(", ")}`).join(" · ")}</p>` : ""}`;
}
async function processDirectNlp() {
  $("#nlpIssues").hidden = true;
  $("#nlpProgress").hidden = false;
  $("#nlpProgressSteps").innerHTML = "";
  try {
    await waitForStep("Removing blank and duplicate records");
    const seen = new Set();
    nlpRawRows = nlpRawRows.filter(value => value.trim() && !seen.has(value.trim().toLowerCase()) && seen.add(value.trim().toLowerCase()));
    await waitForStep("Removing HTML, punctuation, and irrelevant symbols");
    nlpRows = nlpRawRows.map(value => value.replace(/<[^>]*>/g, " ").replace(/[^A-Za-z\s']/g, " ").replace(/\s+/g, " ").toLowerCase().trim());
    await waitForStep("Tokenizing text and removing stopwords");
    nlpRows = nlpRows.map(value => value.split(/\s+/).filter(word => word && !NLP_STOPWORDS.has(word)).join(" "));
    await waitForStep("Creating language features");
    nlpIssuesFixed = true;
    renderNlpOutput();
    renderModelComparison(MODEL_CONFIGS[activeNlpTask]);
    runAllRequested = false;
    $("#comparisonNote").textContent = "Reference estimates are shown below; these scores were not trained on the uploaded file.";
    toast("Issues fixed. NLP analysis complete — insights are ready.");
  } catch (error) {
    toast(`Could not analyse file: ${error.message}`);
  } finally {
    $("#nlpProgress").hidden = true;
  }
}
async function runDirectNlp(file) {
  try {
    nlpFileReady = false;
    nlpIssuesFixed = false;
    nlpRows = [];
    $("#nlpResults").hidden = true;
    const raw = await file.text();
    let extracted;
    if (/\.(xls|xlsx)$/i.test(file.name)) {
      const workbook = XLSX.read(await file.arrayBuffer(), { type: "array" });
      const matrix = XLSX.utils.sheet_to_json(workbook.Sheets[workbook.SheetNames[0]], { header: 1, defval: "" });
      extracted = matrix.slice(1).map(row => row.map(value => String(value)).join(" "));
    } else if (/\.csv$/i.test(file.name)) {
      extracted = parseCSV(raw).slice(1).map(row => row.join(" "));
    } else {
      extracted = raw.split(/\r?\n/);
    }
    nlpRawRows = extracted;
    nlpFileReady = true;
    showNlpFileStatus(file.name, extracted.length);
    const issues = inspectNlpIssues(extracted);
    showNlpIssues(issues);
    if (!issues.blankRecords.length && !issues.duplicateCount) processDirectNlp();
    else toast("File loaded. Review and fix the detected issues before analysis.");
  } catch (error) {
    toast(`Could not read file: ${error.message}`);
  }
}
$$(".nav-item[data-section]").forEach(item => item.addEventListener("click", () => showSection(item.dataset.section)));
$("#filterInput").addEventListener("input", event => { clearTimeout(filterTimer); filterTimer = setTimeout(() => renderTable(event.target.value), 120); });
$("#clearFilter").addEventListener("click", () => { $("#filterInput").value = ""; renderTable(); });
if ($("#buildAnalysis")) $("#buildAnalysis").addEventListener("click", buildAnalysis);
$("#fullscreenPygwalker").addEventListener("click", () => {
  const frame = $("#pygwalkerFrame");
  if (!frame.hidden && frame.requestFullscreen) frame.requestFullscreen();
  else toast("Load the visual explorer before opening full screen.");
});
$("#fillMissing").addEventListener("click", () => {
  if (!preparedRows) preparedRows = rows.map(row => ({ ...row }));
  headers.forEach(header => {
    const values = preparedRows.map(row => row[header]).filter(value => value !== "" && value !== null && value !== undefined);
    if (!values.length) return;
    const numeric = values.every(isNumericValue);
    const replacement = numeric ? values.map(Number).sort((a, b) => a - b)[Math.floor(values.length / 2)] : values.sort((a, b) => values.filter(value => value === a).length - values.filter(value => value === b).length).at(-1);
    preparedRows.forEach(row => { if (row[header] === "" || row[header] === null || row[header] === undefined) row[header] = replacement; });
  });
  markPreparedChange("Missing values filled using median/mode guidance.");
});
$("#removeDuplicates").addEventListener("click", () => {
  if (!preparedRows) preparedRows = rows.map(row => ({ ...row }));
  const unique = new Map(preparedRows.map(row => [JSON.stringify(row), row]));
  preparedRows = [...unique.values()];
  markPreparedChange("Duplicate records removed.");
});
$("#normalizeData").addEventListener("click", () => {
  if (!preparedRows) preparedRows = rows.map(row => ({ ...row }));
  headers.forEach(header => {
    const values = preparedRows.map(row => Number(row[header])).filter(value => !Number.isNaN(value));
    if (values.length < 2 || values.length !== preparedRows.length) return;
    const min = Math.min(...values), max = Math.max(...values);
    preparedRows.forEach(row => { row[header] = max === min ? 0 : (Number(row[header]) - min) / (max - min); });
  });
  markPreparedChange("Numeric fields normalized to a 0–1 range.");
});
$("#encodeCategories").addEventListener("click", () => {
  if (!preparedRows) preparedRows = rows.map(row => ({ ...row }));
  headers.forEach(header => {
    const values = preparedRows.map(row => row[header]);
    if (!values.length || values.every(isNumericValue)) return;
    const dictionary = new Map([...new Set(values)].map((value, index) => [value, index]));
    preparedRows.forEach(row => { row[header] = dictionary.get(row[header]); });
  });
  markPreparedChange("Categorical fields encoded as numeric categories.");
});
$("#cleanText").addEventListener("click", () => applyTextTransform("Text cleaned: HTML, punctuation, numbers, and excess whitespace removed.", value => value.replace(/<[^>]*>/g, " ").replace(/[^A-Za-z\s']/g, " ").replace(/\s+/g, " ").trim().toLowerCase()));
$("#tokenizeText").addEventListener("click", () => applyTextTransform("Text tokenized into normalized word tokens.", value => value.trim().split(/\s+/).filter(Boolean).join(" ")));
$("#removeStopwords").addEventListener("click", () => applyTextTransform("Stopwords removed from text fields.", value => value.split(/\s+/).filter(word => !NLP_STOPWORDS.has(word.toLowerCase())).join(" ")));
$("#stemText").addEventListener("click", () => applyTextTransform("Words reduced to conservative approximate stems.", value => value.split(/\s+/).map(word => word.length > 5 ? word.replace(/(ing|ed|es|s)$/i, "") : word).join(" ")));
$("#expandContractions").addEventListener("click", () => applyTextTransform("Contractions expanded for consistent text features.", value => value.replace(/\b[\w']+\b/g, word => CONTRACTIONS[word.toLowerCase()] || word)));
$("#handleEmoji").addEventListener("click", () => applyTextTransform("Common emojis converted to descriptive text tokens.", value => value.replace(/😀|😃|😄|😁/g, " happy ").replace(/😢|😭/g, " sad ").replace(/❤️|❤|😍/g, " love ").replace(/👍/g, " approve ").replace(/👎/g, " disapprove ").replace(/😂/g, " laugh ")));
$("#nlpGuidance").addEventListener("click", () => toast("For high-quality NLP, review spelling corrections and use POS tagging or named-entity recognition when grammar, people, places, organisations, or dates matter. These advanced steps require a language model and should be validated before training."));
$("#vectorizeText").addEventListener("click", () => {
  if (!preparedRows) preparedRows = rows.map(row => ({ ...row }));
  const fields = updateTextBadge(preparedRows);
  if (!fields.length) { toast("No text fields were detected in the current data."); return; }
  const tokenSets = fields.map(field => preparedRows.map(row => String(row[field] ?? "").toLowerCase().match(/[a-z]{2,}/g) || []));
  tokenSets.forEach((documents, fieldIndex) => {
    const documentFrequency = new Map();
    documents.forEach(tokens => [...new Set(tokens)].forEach(token => documentFrequency.set(token, (documentFrequency.get(token) || 0) + 1)));
    const vocabulary = [...documentFrequency.entries()].sort((a, b) => b[1] - a[1]).slice(0, 20).map(([token]) => token);
    vocabulary.forEach(token => {
      const column = `${fields[fieldIndex]} · tfidf · ${token}`;
      if (!headers.includes(column)) headers.push(column);
      const idf = Math.log((documents.length + 1) / (1 + documentFrequency.get(token))) + 1;
      preparedRows.forEach((row, rowIndex) => {
        const tokens = documents[rowIndex];
        row[column] = tokens.length ? Number((tokens.filter(item => item === token).length / tokens.length * idf).toFixed(4)) : 0;
      });
    });
  });
  markPreparedChange("Text vectorized into TF-IDF features.");
});
$("#trainSplit").addEventListener("input", event => { const train = Number(event.target.value); $("#splitSummary").textContent = `${train} / ${Math.floor((100 - train) / 2)} / ${100 - train - Math.floor((100 - train) / 2)}`; });
$("#applySplit").addEventListener("click", () => {
  const train = Number($("#trainSplit").value); const validation = Math.floor((100 - train) / 2); const test = 100 - train - validation;
  splitInfo = { train, validation, test }; preparationChanges.push(`Data split ${train}/${validation}/${test}`); assessReadiness(); toast(`Split recorded: ${train}% train, ${validation}% validation, ${test}% test.`);
});
$("#augmentData").addEventListener("click", () => {
  if (!preparedRows) preparedRows = rows.map(row => ({ ...row }));
  const numeric = headers.find(header => preparedRows.every(row => isNumericValue(row[header])));
  if (!numeric) { toast("No fully numeric field is available for safe jitter augmentation."); return; }
  const additions = preparedRows.slice(0, Math.min(100, preparedRows.length)).map(row => ({ ...row, [numeric]: Number(row[numeric]) * (1 + (Math.random() - 0.5) * 0.02) }));
  preparedRows.push(...additions); markPreparedChange(`Added ${additions.length} conservative numeric augmentation records.`);
});
$("#annotationGuide").addEventListener("click", () => toast("Best practice: define a label schema, annotate a representative sample, and review inter-annotator agreement before training."));
$("#savePrepared").addEventListener("click", () => {
  if (!preparedRows) preparedRows = rows.map(row => ({ ...row }));
  window.signalMLPreparedData = { headers: [...headers], rows: preparedRows.map(row => ({ ...row })), split: splitInfo };
  $("#datasetName").textContent = `${datasetLabel} · prepared`;
  toast("Prepared data saved for Model Lab.");
});
["pivotRows", "pivotColumns", "pivotMeasure", "pivotAggregation"].forEach(id => { if ($(`#${id}`)) $(`#${id}`).addEventListener("change", buildAnalysis); });
$$(".view-tab").forEach(tab => tab.addEventListener("click", () => {
  $$(".view-tab").forEach(item => item.classList.toggle("active", item === tab));
  $("#chartView").classList.toggle("hidden", tab.dataset.view !== "chart");
  $("#pivotView").classList.toggle("hidden", tab.dataset.view !== "pivot");
}));
$$(".task-option[data-task]").forEach(button => button.addEventListener("click", () => {
  $$(".task-option").forEach(option => option.classList.remove("selected")); button.classList.add("selected"); activeTask = button.dataset.task;
  if (Object.prototype.hasOwnProperty.call(activeSubtasks, activeTask)) activeSubtasks[activeTask] = null;
  renderAdvisorCopy();
  showSecondaryTaskPicker();
  taskSelectionComplete = false;
  setModelAdvisorVisible(false);
  $("#nlpResults").hidden = activeTask !== "nlp" || !nlpFileReady || !nlpIssuesFixed;
  $("#nlpUpload").hidden = true;
  $("#nlpProgress").hidden = true;
  $("#modelResults").hidden = true;
  selectedModel = null;
  renderModelChoices();
  if (activeTask !== "nlp") {
    $("#nlpFileStatus").hidden = true;
    $("#nlpIssues").hidden = true;
    $("#nlpResults").hidden = true;
  }
}));
$$("#nlpTaskPicker .task-option").forEach(button => button.addEventListener("click", () => {
  $$("#nlpTaskPicker .task-option").forEach(option => option.classList.remove("selected"));
  button.classList.add("selected");
  activeNlpTask = button.dataset.nlpTask;
  taskSelectionComplete = true;
  setModelAdvisorVisible(true);
  $("#modelResults").hidden = true;
  selectedModel = null;
  renderModelChoices();
  renderAdvisorCopy();
  renderNlpOutput();
}));
$$(".secondary-task-picker:not(#nlpTaskPicker) .task-option").forEach(button => button.addEventListener("click", () => {
  const picker = button.closest(".secondary-task-picker");
  picker.querySelectorAll(".task-option").forEach(option => option.classList.remove("selected"));
  button.classList.add("selected");
  const pickerId = button.closest(".secondary-task-picker").id;
  const task = pickerId.replace("TaskPicker", "");
  if (Object.prototype.hasOwnProperty.call(activeSubtasks, task)) activeSubtasks[task] = button.dataset.subtask;
  taskSelectionComplete = true;
  setModelAdvisorVisible(true);
  renderModelChoices();
  renderAdvisorCopy();
}));
$("#runModels").addEventListener("click", requestModelComparison); $("#startRecommended").addEventListener("click", () => { showSection("models"); requestModelComparison(); });
$("#nlpFileInput").addEventListener("change", event => { const file = event.target.files[0]; if (file) runDirectNlp(file); event.target.value = ""; });
$("#fixNlpIssues").addEventListener("click", processDirectNlp);
$("#newProject").addEventListener("click", () => toast("New project space created."));
$("#exportReport").addEventListener("click", () => toast("Report export is ready to connect to your chosen format."));
function openTabularFile(file) {
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    try {
      let matrix;
      if (/\.(xls|xlsx)$/i.test(file.name)) {
        const workbook = XLSX.read(reader.result, { type: "array", cellDates: true });
        const firstSheet = workbook.Sheets[workbook.SheetNames[0]];
        matrix = XLSX.utils.sheet_to_json(firstSheet, { header: 1, defval: "" });
      } else {
        matrix = parseCSV(reader.result);
      }
      if (!matrix.length || !matrix[0].length) throw new Error("This file does not contain any tabular data.");
      openPreview(file, matrix);
    } catch (error) {
      toast(`Could not read file: ${error.message}`);
    }
  };
  if (/\.(xls|xlsx)$/i.test(file.name)) reader.readAsArrayBuffer(file); else reader.readAsText(file);
}
$("#modelFileInput").addEventListener("change", event => {
  const file = event.target.files[0];
  closeModelUpload();
  if (file) {
    if (activeTask === "nlp") runDirectNlp(file);
    else openTabularFile(file);
  }
  event.target.value = "";
});
$("#closeModelUpload").addEventListener("click", closeModelUpload);
$("#fileInput").addEventListener("change", event => openTabularFile(event.target.files[0]));
["dragenter","dragover"].forEach(eventName => $("#dropZone").addEventListener(eventName, event => { event.preventDefault(); $("#dropZone").style.borderColor = "var(--violet)"; }));
$("#dropZone").addEventListener("dragleave", () => { $("#dropZone").style.borderColor = ""; });
$("#dropZone").addEventListener("drop", event => { event.preventDefault(); $("#dropZone").style.borderColor = ""; const file = event.dataTransfer.files[0]; if (file && /\.(csv|xls|xlsx)$/i.test(file.name)) { $("#fileInput").files = event.dataTransfer.files; $("#fileInput").dispatchEvent(new Event("change")); } else toast("Please drop a CSV or Excel file."); });
$$("#previewModal input[name='headerChoice']").forEach(input => input.addEventListener("change", () => { collectPreviewSettings(); renderPreview(); }));
$("#loadFull").addEventListener("click", loadPreview);
$("#loadSample").addEventListener("click", loadSample);
$("#closePreview").addEventListener("click", closePreview);
$("#cancelPreview").addEventListener("click", closePreview);
renderModelChoices();
showSecondaryTaskPicker();
setModelAdvisorVisible(false);
updateDatasetUI();
