# Student Placement Prediction System — Demo Prototype

## Important note
The internship report describes a model trained using `placement.csv`, but the PDF does not contain the original CSV or a saved trained model. This app therefore creates a synthetic demonstration dataset and trains a Random Forest model on it when the app starts. It is suitable for demonstrating the interface and workflow, but its predictions and demo accuracy are **not** the actual internship model results.

## Run on Windows (VS Code)

1. Install Python 3.11 or 3.12.
2. Extract this folder.
3. Open the folder in VS Code.
4. Open Terminal → New Terminal.
5. Run:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

6. Open the local URL shown in the terminal (usually `http://localhost:8501`).

## Suggested screenshots for the PPT
- Prediction tab with student inputs and the result visible.
- Dataset & Model tab showing the dataset preview and metrics.
- Dataset & Model tab showing the placement distribution chart and confusion matrix.
- About tab showing the project workflow and technologies.

Use your own running application screenshots. Clearly label this as a demonstration prototype until the original internship dataset/model is available.
