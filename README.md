
# Credit Card Default Prediction

This project predicts credit card default using machine learning models. It includes a Flask API backend and a simple HTML/JS frontend.

## Project Structure

```text
project-root/
│
├── backend/
│   └── app.py                # Flask API
│
├── frontend/
│   ├── index.html            # Simple HTML frontend
│   └── script.js             # JS to call API
│
├── models/
│   ├── best_model.pkl        # Trained model
│   └── scaler_ann.pkl        # Scaler for features
│
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
└── CreditCardFraudDetection_ML.ipynb # Notebook for training
```

## Backend (API)

- **Framework:** Flask
- **Endpoint:** `/predict` (POST)
- **Input:** JSON with `features` (list of feature values)
- **Output:** JSON with `prediction` (0 = No Default, 1 = Default)

## Frontend

- **index.html:** Simple form to collect feature values and display prediction.
- **script.js:** Sends POST request to API and displays result.

## Deployment

- **Backend:** Deploy on [Render](https://render.com/) or [Railway](https://railway.app/) (free tier, supports Docker)
- **Frontend:** Deploy on [Netlify](https://www.netlify.com/) or [GitHub Pages](https://pages.github.com/) (free)

### Docker & HuggingFace

- You can containerize your backend with Docker and deploy to Render or Railway.
- HuggingFace Spaces is not suitable for Flask APIs, but works for Streamlit/Gradio UIs.
- For full-stack Docker, use Docker Compose and deploy to Render.

## Usage

1. Train your model and save `best_model.pkl` and `scaler_ann.pkl` in the `models/` folder.
2. Deploy the backend (`backend/app.py`) using your preferred platform.
3. Deploy the frontend (`frontend/`) as a static site.
4. Update the API URL in `frontend/script.js` after backend deployment.

---

**Notes:**

- Add all required input fields in `index.html` and update the `features` array in `script.js` to match your model's input order.
- For advanced UI, consider using Streamlit or React.

---

**Author:** Om Choksi