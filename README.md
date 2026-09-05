# SignalML

SignalML is a dependency-free first slice of an explainable analytics workspace. It runs entirely in the browser, so uploaded CSV data is not sent anywhere.

## Run locally

Open `index.html` in a browser, or serve the folder with any static web server:

```powershell
python -m http.server 8000
```

Then visit `http://localhost:8000`.

## Included in this first slice

- Guided overview with model recommendation and confidence
- CSV, XLS, and XLSX upload by file picker or drag and drop
- Header detection choice with an editable headerless-file preview
- Preview-time column selection so unneeded fields can be excluded before loading
- Local slice-and-dice workbench with configurable grouping, aggregation, charts, and pivot tables
- Large-file safeguards: responsive table rendering, debounced filtering, category limits, and random record sampling
- Embedded Pygwalker visual exploration through the local Flask service in `pygwalker_backend.py`, with optional browser full-screen mode
- Guided ML-readiness preparation: cleaning, exploration, preprocessing, splitting, augmentation guidance, annotation guidance, and saving prepared data for Model Lab

## Start Pygwalker

Install the Python dependencies and start the local service:

```powershell
py -m pip install -r requirements.txt
py pygwalker_backend.py
```

Then use **Open Pygwalker** in Explore Data. The currently loaded columns and records are sent only to `127.0.0.1`.
- Data table filtering and lightweight column profiling
- Task switcher for classification, regression, clustering, and NLP
- Explainable model comparison visualization with linked scikit-learn documentation
- Automated-preparation narrative and plain-language insights

The model comparison currently uses representative browser-side demo scores. The next implementation step is to add a Python API (for real preprocessing, training, confusion matrices, ROC curves, and downloadable reports) behind this interface.

## Free hosting with Render

The included `render.yaml` deploys the complete app as one Render web service. The Flask service serves the frontend and the visualisation endpoint from the same public URL.

1. Create a GitHub repository and upload this project, including `index.html`, `styles.css`, `app.js`, `pygwalker_backend.py`, `requirements.txt`, `package.json`, `package-lock.json`, and `render.yaml`.
2. In Render, choose **New > Blueprint** and connect the repository.
3. Select the `render.yaml` blueprint and deploy.
4. Open the generated `https://...onrender.com` URL.

The free service may sleep after inactivity and can take a short time to wake up. Uploaded data is processed in memory by the running service and is not saved by SignalML.
