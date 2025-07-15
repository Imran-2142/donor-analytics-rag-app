# 📊 Donor Analytics RAG App

An AI-powered donor insights chatbot that enables users to query donation data using natural language. Built with a **Retrieval-Augmented Generation (RAG)** architecture using LangChain, FAISS, Mistral 7B (via Ollama), and a PostgreSQL database — all wrapped in a user-friendly Streamlit dashboard.

---

## 🚀 Features

- 💬 Natural language Q&A over donor and donation records
- 🧠 Embedding-based search using Sentence Transformers + FAISS
- 🔗 Integrated with LangChain and Mistral 7B for response generation
- 📊 Visual analytics via Streamlit and Plotly
- ✅ Built-in data quality validation and donor segmentation
- 🔒 100% local, privacy-first architecture using Ollama

---

## 🧱 Tech Stack

- **Python**  
- **LangChain**  
- **FAISS** (Vector search)  
- **Sentence Transformers** (`all-MiniLM-L6-v2`)  
- **LLM**: Mistral 7B via [Ollama](https://ollama.com)  
- **PostgreSQL**  
- **Streamlit**  
- **Plotly**

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Imran-2142/donor-analytics-rag-app.git
cd donor-analytics-rag-app


2. Create and Activate Virtual Environment

python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt


3. Configure Environment Variables
Create a .env file in the root directory with the following (or modify as needed):


DB_HOST=localhost
DB_PORT=5432
DB_USER=postgres
DB_PASSWORD=yourpassword
DB_NAME=donor_analytics

CSV_FILE=DonationData_anonymized.csv

OLLAMA_MODEL=mistral:7b
OLLAMA_BASE_URL=http://localhost:11434



4. Set Up the Database
Make sure PostgreSQL is running, then:

python setup_database.py



Run the App
Make sure Ollama is running with the Mistral model:

ollama run mistral

Then launch the Streamlit app:

streamlit run app.py


🧪 Example Questions to Try
"How many donors do we have?"

"What is the total amount raised in 2024?"

"List the top 5 donors by lifetime value"

"Segment our donor base by engagement"

"What’s the average donation this year?"




📁 Project Structure
bash
Copy
Edit
.
├── setup_database.py         # Set up PostgreSQL DB from CSV
├── enhanced_rag_system.py    # Main RAG + validation logic
├── app.py                    # Streamlit app
├── database_verification.py  # Integrity checks and summaries
├── DonationData_anonymized.csv
├── .env                      # (Not committed) DB + API config
├── requirements.txt
└── README.md
📊 Dashboard Preview
<img src="screenshots/dashboard.png" width="100%"/>




What I Learned
Building full-stack RAG pipelines with vector search + LLMs

Using LangChain and Ollama for LLM orchestration

Performing automated data validation and segmentation

Deploying end-to-end AI solutions with real datasets




Acknowledgements
Built as part of my internship at Empact Pte Ltd, Singapore, 2025.


















