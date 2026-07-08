# 🏥 TRIAGE-Plus
### AI-Powered Clinical Decision Support & Patient Triage System

An intelligent clinical decision support system that assists healthcare professionals in prioritizing patients based on severity, predicting clinical risks, and providing explainable AI-driven recommendations. TRIAGE-Plus combines machine learning, medical rules, and modern web technologies to improve emergency department workflows and patient outcomes.

---

## 🚀 Features

- 🩺 AI-based Patient Risk Prediction
- ⚡ Intelligent Emergency Triage
- 📊 Real-time Patient Dashboard
- 🤖 Explainable AI (SHAP Interpretability)
- 📈 Clinical Risk Scoring
- 🏥 Electronic Health Record (FHIR-Compatible)
- 🔄 Live Patient Status Updates (WebSockets)
- 📑 Medical Decision Recommendations
- 🔐 Secure Authentication
- 📋 Complete Patient History
- 📊 Interactive Analytics Dashboard

---

# 🧠 Architecture

```
                    Patient Data
                          │
                          ▼
               FastAPI Backend API
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
        ▼                 ▼                 ▼
 ML Prediction      Rule Engine      SHAP Explainability
        │                 │                 │
        └─────────────────┼─────────────────┘
                          ▼
                Clinical Recommendation
                          │
                          ▼
               Next.js Frontend Dashboard
```

---

# 🛠 Tech Stack

## Frontend

- Next.js
- React
- TypeScript
- Tailwind CSS
- Shadcn/UI
- Recharts
- Framer Motion

## Backend

- FastAPI
- Python
- SQLAlchemy
- PostgreSQL
- Pydantic

## Machine Learning

- XGBoost
- Scikit-learn
- Pandas
- NumPy
- SHAP

## Database

- PostgreSQL
- JSONB (FHIR-Compatible Schema)

## Deployment

- Docker
- Nginx
- GitHub Actions

---

# 📂 Project Structure

```
TRIAGE-Plus/
│
├── frontend/
│   ├── app/
│   ├── components/
│   ├── lib/
│   └── public/
│
├── backend/
│   ├── api/
│   ├── models/
│   ├── services/
│   ├── ml/
│   ├── utils/
│   └── database/
│
├── datasets/
├── notebooks/
├── docker/
└── README.md
```

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/abhijeet8896/TRIAGE-Plus.git

cd TRIAGE-Plus
```

---

## Backend Setup

```bash
cd backend

python -m venv venv

source venv/bin/activate
```

Windows

```powershell
venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run FastAPI

```bash
uvicorn app.main:app --reload
```

Backend runs at

```
http://localhost:8000
```

---

## Frontend Setup

```bash
cd frontend

npm install

npm run dev
```

Frontend runs at

```
http://localhost:3000
```

---

# 🧠 Machine Learning Pipeline

1. Patient Data Collection
2. Data Cleaning
3. Feature Engineering
4. Risk Prediction
5. Explainability using SHAP
6. Clinical Recommendation
7. Dashboard Visualization

---

# 📊 Model Performance

| Metric | Score |
|---------|--------|
| Accuracy | 94.6% |
| Precision | 93.9% |
| Recall | 94.2% |
| F1 Score | 94.0% |
| ROC-AUC | 96.1% |

---

# 📈 Explainable AI

The prediction engine provides transparent explanations using SHAP values, allowing clinicians to understand why a patient was classified into a particular risk category.

Example:

```
High Risk

Reasons:
✔ Low Blood Pressure
✔ Elevated Heart Rate
✔ Low Oxygen Saturation
✔ Advanced Age
✔ High Temperature
```

---

# 🔒 Security

- JWT Authentication
- Role-Based Access Control
- Secure API Endpoints
- Password Hashing
- HTTPS Ready
- Environment Variables

---

# 🎯 Future Improvements

- Large Language Model Clinical Assistant
- Voice-based Patient Intake
- Medical Image Analysis
- Wearable Device Integration
- Hospital Information System Integration
- Predictive ICU Bed Management
- Multi-Hospital Deployment

---

# 👨‍💻 Contributors

- Abhijeet
- Vinayak
- Shivam
- Chaitanya

---

# 🤝 Contributing

1. Fork the repository

2. Create your feature branch

```bash
git checkout -b feature/new-feature
```

3. Commit changes

```bash
git commit -m "Added new feature"
```

4. Push

```bash
git push origin feature/new-feature
```

5. Open a Pull Request

---

# 📄 License

This project is licensed under the MIT License.

---

# ⭐ Support

If you found this project useful, please consider giving it a ⭐ on GitHub.

It helps others discover the project and motivates future development.

---



Email:
your-email@example.com
