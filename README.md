# Donor Analytics RAG System - Real Data Edition

An AI-powered donor management chatbot using your real donation data with 2,456+ records.

## 🎯 What This System Does

- Analyzes your real donation database with natural language questions
- Converts questions like "Who is our top donor?" into SQL queries automatically
- Provides intelligent answers with data visualizations
- Works 100% locally for complete data privacy

## 📊 Your Data Structure

- **2,456 donation records** from your CSV file
- **Real donor information** with anonymized IDs
- **Date range:** 2005-2010+ historical data
- **Channels:** Giro, Cash, Cheque payments
- **Types:** Tax and Non-Tax donations
- **Demographics:** Age ranges, donor types

## 🚀 Quick Setup

### 1. Prerequisites
- Install PostgreSQL (remember password!)
- Install Ollama from https://ollama.ai/download
- Download AI model: `ollama pull mistral:7b`

### 2. Setup in PyCharm
1. Create new Python project with virtual environment
2. Copy `DonationData_anonymized.csv` to project folder
3. Create the 6 files shown above
4. Update `.env` with your PostgreSQL password

### 3. Install & Run
```bash
pip install -r requirements.txt
python setup_database.py
streamlit run app.py