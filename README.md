# 🚗 ASPM — Accident Severity Prediction Model

ASPM (Accident Severity Prediction Model) is a **machine learning–based accident risk prediction platform** built with **Plotly Dash**, **Bootstrap 5**, and a **Neumorphism-inspired UI**.  
It predicts accident severity using real crash data (2021–2023) and generates **AI-powered government safety recommendations** through the **Google Gemini API**.

🌐 **Live Demo:** [http://159.89.193.70:8080](http://159.89.193.70:8080)

---

## 🧩 Features

- 🔍 **Accident Severity Prediction** using trained ML model (`aspm_model.pkl`)
- 🧠 **AI Recommendations** from **Google Gemini 2.5 Flash API**
- 🎨 **Modern Neumorphism UI** built with Dash + Bootstrap
- 🌐 **Multi-page Layout** — Landing, Prediction, Visualization
- 🐳 **Dockerized Deployment** on **DigitalOcean Droplets**
- 📊 **Interactive Visualizations** for accident data analysis
- 🔒 **Secure Environment Variables** via `.env`
- ⚡ **Cross-Platform Compatibility** (AMD64 / ARM64)

---

## 🏗️ Project Structure

```
ASPM/
│
├── app.py                  # Dash app entrypoint (Lambda-compatible handler)
├── pages/
│   ├── landing.py          # Landing page (overview visuals)
│   ├── prediction.py       # ML + AI recommendation page
│   ├── visualization.py    # Data visualization dashboard
│
├── models/
│   ├── aspm_model.pkl
│   ├── aspm_scaler.pkl
│   ├── target_encoder.pkl
│
├── data/
│   ├── final_crash_data_2021_2023.csv
│   ├── ...
│
├── requirements.txt
├── pyproject.toml
├── uv.lock
├── Dockerfile
├── .env                    # Environment variables (ignored via .gitignore)
└── README.md
```

---

## ⚙️ Setup & Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/mastersubhajit/aspm.git
cd aspm
```

### 2️⃣ Create a Virtual Environment
Using **uv (recommended)**:
```bash
uv venv .venv
source .venv/bin/activate
uv sync
```

Or using **pip**:
```bash
python -m venv venv
source venv/bin/activate  # (venv\Scripts\activate on Windows)
pip install -r requirements.txt
```

### 3️⃣ Configure Environment Variables
Create a `.env` file in the project root:
```
GOOGLE_API_KEY=your_google_gemini_api_key
```

> 🔐 The `.env` file is automatically excluded from version control.

---

## 🧠 Model Details

- Dataset: **U.S. CRSS Crash Data (2021–2023)**
- Target variable: `MAX_SEVNAME` (accident severity)
- Type: Multi-class classification (Low, Medium, High Risk)
- Accuracy: **>90% across all classes**
- Stored in `/models/aspm_model.pkl` and loaded dynamically

---

## 🧰 Running Locally

```bash
python app.py
```

Then open your browser and go to:
```
http://127.0.0.1:8080
```

To use a custom port:
```bash
PORT=8050 python app.py
```

---

## 🐳 Docker Deployment (DigitalOcean Droplet)

### 1️⃣ Build Docker Image
```bash
docker build -t aspm:latest .
```

### 2️⃣ Run the Container
```bash
docker run -d -p 8080:8080 --env-file .env aspm:latest
```

Then visit your deployed instance:  
👉 [http://159.89.193.70:8080](http://159.89.193.70:8080)

### 3️⃣ Optional: Auto-Restart on Reboot
```bash
docker run -d   --name aspm   --restart unless-stopped   -p 8080:8080   --env-file .env   aspm:latest
```

---

## ☁️ Deploying on DigitalOcean Droplets

1. SSH into your droplet:
   ```bash
   ssh root@<your-droplet-ip>
   ```

2. Install Docker:
   ```bash
   curl -fsSL https://get.docker.com -o get-docker.sh
   sh get-docker.sh
   ```

3. Pull pre-built image (optional):
   ```bash
   docker pull mastersubhajit/aspm:latest
   ```

4. Run the container:
   ```bash
   docker run -d -p 8080:8080 --env-file .env mastersubhajit/aspm:latest
   ```

Your app will be available at:  
👉 [http://159.89.193.70:8080](http://159.89.193.70:8080)

---

## 📊 Visualization Dashboard

The **Visualization** page provides:
- Crash severity distribution by year
- Impact of lighting, speed limits, and road types
- Vehicle type vs. injury outcomes
- Interactive, filterable data graphs powered by Plotly

---

## 🤖 AI Recommendation Engine

ASPM uses **Google Gemini 2.5 Flash API**  
(`https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent`)  
to deliver **evidence-based government safety recommendations** derived from accident context.

If the API is unavailable, a **context-aware fallback engine** provides local recommendations.

---

## 🧱 Technologies Used

| Category | Technology |
|-----------|-------------|
| Frontend | Dash, Plotly, Bootstrap 5, Neumorphism CSS |
| Backend | Flask (via Dash) |
| Machine Learning | Scikit-learn, Pandas, NumPy |
| AI | Google Gemini 2.5 Flash |
| Deployment | Docker + DigitalOcean Droplet |
| Package Manager | uv (Astral) |
| Dataset | NHTSA CRSS 2021–2023 Crash Dataset |

---

## 🛡️ Security

- `.env` securely stores API keys and secrets  
- `.gitignore` excludes sensitive files  
- No hardcoded credentials in the codebase  
- Uses `os.getenv()` for runtime variable access  

---

## 🪄 License

Licensed under the **MIT License**.  
See the [LICENSE](LICENSE) file for details.

---

## ⭐️ Support

If this project helps your research or deployment, please ⭐️ the repository!  

```bash
git clone https://github.com/mastersubhajit/aspm.git
```

---

## 🚀 Quick Links

| Resource | Link |
|-----------|------|
| 🌐 **Live App** | [http://159.89.193.70:8080](http://159.89.193.70:8080) |
| 🐋 **Docker Hub** | [mastersubhajit/aspm](https://hub.docker.com/r/mastersubhajit/aspm) |
| 💻 **GitHub Repo** | [ASPM](https://github.com/mastersubhajit/aspm) |
| 📊 **Dataset** | Private (CRSS 2021–2023, NHTSA) |
