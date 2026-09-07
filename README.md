# Skin Lesion Classifier — Website + Model API

Two parts run side by side:
  • PHP website (XAMPP / Apache, port 80) — the upload page
  • Python API (Flask + TensorFlow, port 5000) — runs your model

The browser → PHP (predict.php) → Python API (app.py) → back to the browser.

────────────────────────────────────────────────────────────────────────────
FOLDER PLACEMENT
────────────────────────────────────────────────────────────────────────────
1) PYTHON API  — put this folder ANYWHERE OUTSIDE htdocs, e.g.  C:\skin_api\

   C:\skin_api\
   ├── app.py
   ├── model_def.py
   ├── requirements.txt
   ├── cleaning\
   │   ├── __init__.py
   │   └── clean.py
   └── models\
       └── best_model_m2.weights.h5     ←  COPY YOUR DOWNLOADED FILE HERE

   (create the empty `models\` folder and drop your .h5 into it)

2) PHP WEBSITE — put in XAMPP htdocs:

   C:\xampp\htdocs\skincancer\
   ├── index.php
   └── predict.php

────────────────────────────────────────────────────────────────────────────
ONE-TIME SETUP (Python)
────────────────────────────────────────────────────────────────────────────
   cd C:\skin_api
   pip install -r requirements.txt

────────────────────────────────────────────────────────────────────────────
RUNNING IT (every time)
────────────────────────────────────────────────────────────────────────────
   Terminal 1 (Python API):
     cd C:\skin_api
     python app.py
     → wait for "Ready." then it serves on http://localhost:5000

   Then start XAMPP → Apache.

   Open in browser:  http://localhost/skincancer/
   Upload a lesion image → Analyze → see Benign/Malignant + probability.

────────────────────────────────────────────────────────────────────────────
NOTES
────────────────────────────────────────────────────────────────────────────
• Preprocessing matches training exactly: M1+M2 cleaning, then raw [0,255]
  fed to the model (it normalizes internally via include_preprocessing=True).
• THRESHOLD in app.py is 0.16 (your F2-optimal). Use 0.50 for balanced.
• APPLY_CLEANING=True because this is the CLEANED model. If you ever switch to
  a model trained on RAW images, set APPLY_CLEANING=False in app.py.
• You uploaded WEIGHTS only (best_model_m2.weights.h5), so app.py rebuilds the
  architecture (model_def.py) and loads the weights into it — this is why a
  plain load_model() is not used.
• Health check: open http://localhost:5000/health → {"status":"ok"}
• This is a research prototype, NOT a medical device. Keep the disclaimer.
