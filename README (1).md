# Carbon Footprint Classification — Streamlit App

A simple Streamlit web app that predicts a product's carbon footprint class
(High / Medium / Low) using a pre-trained scikit-learn pipeline
(imputation + one-hot encoding + logistic regression).

## Project structure

```
carbon_app/
├── app.py                      # Streamlit app
├── carbon_footprint_model.pkl  # Trained sklearn Pipeline (place here)
├── class_names.pkl             # Class labels (place here)
└── requirements.txt            # Python dependencies
```

## Setup

1. Copy `carbon_footprint_model.pkl` and `class_names.pkl` into this folder
   (same folder as `app.py`).

2. Create a virtual environment (recommended) and install dependencies:

   ```bash
   python -m venv venv
   source venv/bin/activate      # on Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

   **Important:** the model was trained with scikit-learn 1.7.2. Installing a
   different version (e.g. 1.8.x) can cause a hard error
   (`AttributeError: 'SimpleImputer' object has no attribute '_fill_dtype'`)
   when unpickling, not just a warning — so make sure `scikit-learn==1.7.2`
   is installed exactly as pinned in `requirements.txt`.

3. Run the app:

   ```bash
   streamlit run app.py
   ```

4. Open the URL shown in the terminal (usually `http://localhost:8501`).

## How it works

- The app collects 13 product/manufacturing inputs through a simple two-column form
  (product type, material, weight, quantity, transport details, packaging,
  recycled content %, renewable energy %, energy use, and total CO₂e).
- On clicking **Predict Carbon Class**, the inputs are assembled into a
  single-row DataFrame matching the columns the model was trained on, and
  passed through `model.predict()` / `model.predict_proba()`.
- The predicted class and a probability breakdown (table + bar chart) are
  displayed.

## Notes

- `class_names.pkl` isn't required for prediction since `model.classes_`
  already returns the labels (`['High', 'Medium', 'Low']`), but it's loaded
  in `app.py` in case you want to use it elsewhere (e.g. for custom ordering
  or display text).
