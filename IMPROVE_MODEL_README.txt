AgroCare Improved Model
=======================
The current old model still works immediately. To activate the improved disease-recognition features, retrain once on your PlantVillage color dataset.

From this project folder run:

py train_improved_model.py --dataset "E:\dataset_source\color" --max-per-class 300

Wait until it prints: Saved: models/best_model.joblib
Then run:

py -m streamlit run app.py

What changed:
- Keeps KNN + Logistic Regression as required by the project plan.
- Adds richer HSV/Lab color histograms, lesion-color ratios, texture, edge and spatial features.
- KNN uses distance weighting.
- Logistic Regression is tuned and both models are compared automatically.
- The existing old 11-feature model remains compatible until you retrain.
- UI, bilingual mode, weather, database/history and cyber design are unchanged.
