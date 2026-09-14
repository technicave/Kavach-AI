# 🛡️ Kavach-AI — AI-Powered Document Verification & Tampering Detection System

**Kavach-AI** is an advanced artificial intelligence backend designed to detect tampering, forgery, and authenticity issues in official government documents such as Passports, Aadhaar Cards, Driving Licenses, and PAN Cards. Built for high performance, real-time response, and smooth integration with web and mobile frontends.

---

## 🚀 Features

- 📄 **Document Authenticity Check:** Verifies structural and layout integrity of government IDs.
- 🔍 **Tampering & Forgery Detection:** Scans for digital manipulation, font inconsistencies, and altered text/images.
- ⚡ **Fast API Endpoints:** Powered by FastAPI for asynchronous performance and ultra-low latency.
- 📖 **Interactive API Specs:** Auto-generated Swagger UI and ReDoc documentation for seamless developer integration.

---

## 🛠️ Prerequisites

Ensure you have the exact Python version installed before proceeding:

* **Python:** `3.12.14`
* **Pip:** Latest version
* **OS:** Linux / macOS / Windows (WSL recommended for Linux steps)

---

## 📦 Installation & Setup

Follow these steps to set up the development environment and get the backend running locally.

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/kavach-ai.git
cd kavach-ai
```

### 2. Create and Activate Virtual Environment

**On Linux / macOS:**
```bash
python3.12 -m venv env
source env/bin/activate
```

**On Windows (PowerShell):**
```powershell
py -3.12 -m venv env
.\env\Scripts\Activate.ps1
```

### 3. Install Dependencies
Install all required modules specified in the `requirements.txt` file:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🏃 Running the Application

1. **Activate the environment** (if not already activated):
   ```bash
   source env/bin/activate
   ```

2. **Navigate to the backend directory:**
   ```bash
   cd backend
   ```

3. **Start the FastAPI server:**
   ```bash
   uvicorn app:app --reload
   ```

The server will launch and run locally (default: `http://127.0.0.1:8000`).

---

## 📚 API Documentation

FastAPI provides an automatic interactive documentation interface:

* **Swagger UI (Interactive Docs):** Open your browser and visit:  
  `http://127.0.0.1:8000/docs`
* **ReDoc (Alternative Spec View):**  
  `http://127.0.0.1:8000/redoc`

Use these pages to test the API endpoints directly from your browser!

---

## 🧪 Project Structure

```text
.
├── env/                   # Virtual Environment
├── backend/
│   ├── app.py             # Main FastAPI entry point
│   ├── controllers/       # Route handlers & controllers
│   ├── models/            # Machine learning / computer vision models
│   └── utils/             # Helper utilities & image processing logic
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request