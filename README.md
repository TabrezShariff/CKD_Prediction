# CKD Prediction Web App

This is a Chronic Kidney Disease (CKD) prediction web application built using a machine learning model trained on a static dataset (`kidney_data.xlsx`) sourced from Kaggle. The frontend is built with HTML, CSS, and JavaScript, while the backend uses Python's Flask framework.

The application allows users to input relevant medical parameters, predicts the likelihood of CKD, and displays results both numerically and graphically. A **donut graph** visually represents the risk—**higher fill indicates higher CKD risk**.

---

## 🚀 Features

- CKD prediction using a trained machine learning model
- User-friendly web interface
- Real-time result display
- Donut chart representation for visualizing CKD risk

---

## 📊 Dataset

- **Source**: [Kaggle - Chronic Kidney Disease Dataset](https://www.kaggle.com/)
- **Format**: `.xlsx`
- **Usage**: Used for training the model offline. The trained model is then saved and used for real-time predictions.

---

## 🛠️ Tech Stack

| Frontend | Backend | ML & Data | Visualization |
|----------|---------|-----------|----------------|
| HTML     | Flask   | pandas, scikit-learn | Chart.js (for donut graphs) |
| CSS      |         | kidney_data.xlsx     |                |
| JavaScript |       |                    |                |

---

## 💻 How to Run Locally

### 🔧 Prerequisites

- Python 3.x
- pip
- (Recommended) Virtual Environment

## ⚙️ Setup Steps


### 1. Clone the repo

```bash
git clone https://github.com/your-username/ckd-prediction-app.git
cd ckd-prediction-app
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix or Mac
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
python app.py
```

### Once the server starts, open your browser and go to:

```bash
http://127.0.0.1:5000/
```

## 📌 Future Enhancements

- Add support for live API-based predictions
- Store and track patient history
- Deploy on platforms like Heroku or Render

## 🙌 Acknowledgements

- Dataset: Kaggle - Chronic Kidney Disease Dataset
- Libraries: Flask, scikit-learn, pandas, Chart.js

## 📸 Snapshots

![Screenshot 2025-06-14 002024](https://github.com/user-attachments/assets/dbe87121-ece4-4f5d-a79d-963b13bf2660)
![Screenshot 2025-06-14 002109](https://github.com/user-attachments/assets/a6c94fb2-d8da-40c3-aca1-2fc65bdfdfe8)
![Screenshot 2025-06-14 002204](https://github.com/user-attachments/assets/a73b31bc-9285-4faf-85a6-1025fa9a35c3)
![Screenshot 2025-06-14 002414](https://github.com/user-attachments/assets/6a204d74-8e02-450b-bbda-718c355fcd04)
