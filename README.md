# Disease Prediction Project

This repository contains a deployable Streamlit web application for predicting diseases from symptoms using the provided medical dataset.

## Project structure

- `app.py` - Streamlit web interface and prediction logic
- `Training.csv` - Dataset used for training the model
- `requirements.txt` - Python dependencies

## Live Demo

Open the live deployed app here:

https://2ctrwclsxkewkbh5yjwkhf.streamlit.app/

## Run locally

```powershell
pip install -r requirements.txt
py -m streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Push this repository to GitHub.
2. Open [Streamlit Community Cloud](https://streamlit.io/cloud).
3. Click `New app`.
4. Connect the GitHub repository.
5. Set the main file to `app.py`.
6. Deploy.

## Notes

The app is designed to be simple and GitHub-friendly for deployment and showcasing a medical symptom prediction workflow.
