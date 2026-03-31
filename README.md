# Credit Card Default Prediction

This project predicts credit card default using machine learning models. It includes a Flask API backend and a simple HTML/JS frontend.

## Project Structure
```
project-root/
│
├── Models/
│   └── best_model.pkl
│   └── scaler_ann.pkl
│
├── app.py                # Flask API
├── requirements.txt      # Python dependencies
│
└── frontend/
    ├── index.html        # Simple HTML frontend
    └── script.js         # JS to call API
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
- **Backend:** Deploy on Render, Railway, or Deta (free tier)
- **Frontend:** Deploy on GitHub Pages or Netlify (free)

## Usage
1. Train your model and save `best_model.pkl` and `scaler_ann.pkl` in the `Models/` folder.
2. Deploy the backend (`app.py`) using your preferred platform.
3. Deploy the frontend (`frontend/`) as a static site.
4. Update the API URL in `frontend/script.js` after backend deployment.

---

**Note:**
- Add all required input fields in `index.html` and update the `features` array in `script.js` to match your model's input order.
- For advanced UI, consider using Streamlit or React.


Om Choksi