import os
import psycopg2
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Load environment variables
load_dotenv()


class DonorRAGSystem:
    def __init__(self):
        """Initialize the RAG system with database connection and AI models"""
        self.db_params = {
            'host': os.getenv('DB_HOST'),
            'port': os.getenv('DB_PORT'),
            'database': os.getenv('DB_NAME'),
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD')
        }

        # Initialize sentence transformer for embeddings
        print("🔄 Loading sentence transformer model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Initialize Ollama LLM
        print("🔄 Connecting to Ollama...")
        try:
            self.llm = Ollama(
                model=os.getenv('OLLAMA_MODEL', 'mistral:7b'),
                base_url=os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
            )
            # Test connection
            self.llm.invoke("test")
            print("✅ Ollama connection successful!")
        except Exception as e:
            print(f"❌ Ollama connection failed: {e}")
            print("Make sure Ollama is running and mistral:7b model is downloaded")
            raise

        # Initialize FAISS vector store
        self.vector_store = None
        self.documents = []

        # Create embeddings from database
        self.create_vector_store()

        print("✅ RAG System initialized successfully!")

    def get_database_schema(self):
        """Get database schema information for the real data"""
        schema_info = """
        Database Schema for Real Donor Data:

        Table: donors
        - donor_id (VARCHAR): Unique donor identifier (same as donor_name)
        - donor_name (VARCHAR): Donor name (e.g., Donor_001, Donor_002)
        - age_range (VARCHAR): Age categories ('35-45', 'Over 56', etc.)
        - donor_id_number (VARCHAR): ID number 
        - donor_id_type (VARCHAR): ID type (FIN, NRIC, etc.)
        - donor_type (VARCHAR): Individual, Corporation, etc.
        - total_donations (INTEGER): Number of donations made
        - lifetime_value (DECIMAL): Total amount donated by donor
        - first_donation_date (DATE): Date of first donation
        - last_donation_date (DATE): Date of most recent donation

        Table: donations
        - donation_id (VARCHAR): Unique donation identifier (DON_00001, etc.)
        - donor_id (VARCHAR): References donors.donor_id
        - amount (DECIMAL): Donation amount in currency
        - donation_date (DATE): When donation was made
        - channel (VARCHAR): Payment method (Giro, Cash, Cheque, etc.)
        - donation_type (VARCHAR): Tax classification (Tax(Outright Cash), Non-Tax(Outright Cash))
        - donor_tier_name (VARCHAR): Donor tier if applicable
        - donor_tier_amount (DECIMAL): Tier threshold amount
        - donor_tier_description (TEXT): Tier description
        """
        return schema_info

    def get_sample_data(self):
        """Get sample data from each table for context"""
        try:
            conn = psycopg2.connect(**self.db_params)

            # Get sample donors
            donors_df = pd.read_sql("""
            SELECT donor_id, donor_name, age_range, donor_type, 
                   total_donations, lifetime_value
            FROM donors 
            ORDER BY lifetime_value DESC
            LIMIT 5
            """, conn)

            # Get sample donations
            donations_df = pd.read_sql("""
            SELECT d.donation_id, d.donor_id, d.amount, d.donation_date, 
                   d.channel, d.donation_type
            FROM donations d
            WHERE d.amount IS NOT NULL
            ORDER BY d.donation_date DESC NULLS LAST
            LIMIT 5
            """, conn)

            # Get summary statistics
            stats_df = pd.read_sql("""
            SELECT 
                COUNT(DISTINCT donor_id) as total_donors,
                COUNT(*) as total_donations,
                SUM(amount) as total_amount,
                AVG(amount) as avg_donation,
                MIN(donation_date) as earliest_donation,
                MAX(donation_date) as latest_donation
            FROM donations
            WHERE amount IS NOT NULL
            """, conn)

            conn.close()

            return {
                'donors': donors_df,
                'donations': donations_df,
                'stats': stats_df
            }

        except Exception as e:
            print(f"Error getting sample data: {e}")
            return {}

    def create_vector_store(self):
        """Create vector embeddings from database content"""
        print("🔄 Creating vector embeddings from database...")

        try:
            # Get schema and sample data
            schema = self.get_database_schema()
            sample_data = self.get_sample_data()

            # Create documents for embedding
            documents = []

            # Add schema information
            documents.append(f"Database Schema for Real Donor Data:\n{schema}")

            # Add sample data context
            if 'donors' in sample_data and not sample_data['donors'].empty:
                donors_context = f"Sample Top Donors:\n{sample_data['donors'].to_string(index=False)}"
                documents.append(donors_context)

            if 'donations' in sample_data and not sample_data['donations'].empty:
                donations_context = f"Sample Recent Donations:\n{sample_data['donations'].to_string(index=False)}"
                documents.append(donations_context)

            if 'stats' in sample_data and not sample_data['stats'].empty:
                stats_context = f"Database Statistics:\n{sample_data['stats'].to_string(index=False)}"
                documents.append(stats_context)

            # Add query patterns specific to the real data
            query_patterns = """
            PERFECT SQL Query Examples for Real Donor Data:

            1. Top donors by lifetime value (SIMPLE - no JOINs needed):
            SELECT donor_name, lifetime_value FROM donors ORDER BY lifetime_value DESC LIMIT 10;

            2. Total donations by channel:
            SELECT channel, COUNT(*) as donation_count, SUM(amount) as total_amount FROM donations WHERE channel IS NOT NULL GROUP BY channel ORDER BY total_amount DESC LIMIT 10;

            3. Donors by age range:
            SELECT age_range, COUNT(*) as donor_count, SUM(lifetime_value) as total_value FROM donors WHERE age_range IS NOT NULL GROUP BY age_range ORDER BY total_value DESC LIMIT 10;

            4. Donors with more than X donations:
            SELECT d.donor_name, COUNT(*) as donation_count FROM donors d JOIN donations dn ON d.donor_id = dn.donor_id GROUP BY d.donor_name HAVING COUNT(*) > 5 ORDER BY donation_count DESC LIMIT 10;

            5. Recent donations:
            SELECT dn.donation_id, d.donor_name, dn.amount, dn.donation_date, dn.channel FROM donations dn JOIN donors d ON dn.donor_id = d.donor_id ORDER BY dn.donation_date DESC NULLS LAST LIMIT 10;

            6. Tax vs Non-Tax donations:
            SELECT donation_type, COUNT(*) as count, SUM(amount) as total FROM donations WHERE donation_type IS NOT NULL GROUP BY donation_type ORDER BY total DESC;

            CRITICAL RULES:
            - For top donors: USE donors table only (lifetime_value already calculated)
            - For aggregations: Include ALL non-aggregate columns in GROUP BY
            - Always use LIMIT clauses
            - Use table aliases consistently: d for donors, dn for donations
            """
            documents.append(query_patterns)

            self.documents = documents

            # Create embeddings
            embeddings = self.embedding_model.encode(documents)

            # Create FAISS index
            dimension = embeddings.shape[1]
            self.vector_store = faiss.IndexFlatL2(dimension)
            self.vector_store.add(embeddings.astype('float32'))

            print(f"✅ Vector store created with {len(documents)} documents")

        except Exception as e:
            print(f"❌ Error creating vector store: {e}")
            self.documents = []

    def retrieve_context(self, query, k=3):
        """Retrieve relevant context for the query"""
        if not self.vector_store or not self.documents:
            return ""

        try:
            # Create query embedding
            query_embedding = self.embedding_model.encode([query])

            # Search vector store
            distances, indices = self.vector_store.search(query_embedding.astype('float32'), k)

            # Get relevant documents
            relevant_docs = []
            for idx in indices[0]:
                if idx < len(self.documents):
                    relevant_docs.append(self.documents[idx])

            return "\n\n".join(relevant_docs)

        except Exception as e:
            print(f"Error retrieving context: {e}")
            return ""

    def execute_sql_query(self, query):
        """Execute SQL query and return results"""
        try:
            conn = psycopg2.connect(**self.db_params)

            # Execute query
            df = pd.read_sql(query, conn)
            conn.close()

            return df

        except Exception as e:
            print(f"SQL Error: {e}")
            return None

    def generate_sql_from_question(self, question):
        """Generate SQL query from natural language question"""

        # Get relevant context
        context = self.retrieve_context(question)

        # Create prompt for SQL generation
        sql_prompt = PromptTemplate(
            input_variables=["question", "context"],
            template="""
You are a PostgreSQL expert. Generate ONE perfect SQL query to answer the user's question.

Context and Perfect Examples:
{context}

Question: {question}

ABSOLUTE RULES:
1. Generate ONLY ONE clean SQL query - no explanations, no comments
2. For "top donors" questions: USE ONLY donors table (lifetime_value is pre-calculated)
3. For GROUP BY: Include ALL non-aggregate columns in GROUP BY clause
4. Always use LIMIT clauses (5-20 records)
5. Use proper table aliases: d for donors, dn for donations
6. No SQL comments (-- or /* */)
7. End with semicolon

PERFECT PATTERNS:
- Top donors: SELECT donor_name, lifetime_value FROM donors ORDER BY lifetime_value DESC LIMIT 10;
- By channel: SELECT channel, COUNT(*), SUM(amount) FROM donations WHERE channel IS NOT NULL GROUP BY channel LIMIT 10;
- With JOINs: SELECT d.donor_name, COUNT(*) FROM donors d JOIN donations dn ON d.donor_id = dn.donor_id GROUP BY d.donor_name LIMIT 10;

SQL Query:
"""
        )

        try:
            # Use invoke instead of deprecated methods
            response = self.llm.invoke(sql_prompt.format(question=question, context=context))

            # Extract and clean SQL query
            sql_query = response.strip()

            # Remove code blocks
            if "```sql" in sql_query:
                sql_query = sql_query.split("```sql")[1].split("```")[0]
            elif "```" in sql_query:
                sql_query = sql_query.split("```")[1]

            # Remove comments and extra text
            lines = sql_query.split('\n')
            clean_lines = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('--') and not line.startswith(
                        '/*') and not 'explanation' in line.lower():
                    clean_lines.append(line)

            sql_query = ' '.join(clean_lines).strip()

            # Remove any trailing semicolon and add one
            sql_query = sql_query.rstrip(';') + ';'

            return sql_query

        except Exception as e:
            print(f"Error generating SQL: {e}")
            return None

    def generate_answer(self, question, sql_result):
        """Generate natural language answer from SQL results"""

        # Convert DataFrame to string representation
        if sql_result is not None and not sql_result.empty:
            result_text = sql_result.to_string(index=False)
            row_count = len(sql_result)
        else:
            result_text = "No results found."
            row_count = 0

        # Create prompt for answer generation
        answer_prompt = PromptTemplate(
            input_variables=["question", "results", "row_count"],
            template="""
Based on the SQL query results, provide a clear answer to the user's question about donor data.

Question: {question}
Number of results: {row_count}

Query Results:
{results}

Instructions:
1. Provide a direct, helpful answer
2. Include specific numbers, names, and amounts
3. Format currency amounts clearly (e.g., $1,234.56)
4. If multiple results, highlight the top ones
5. Use a professional but friendly tone
6. If no results, suggest why and alternative approaches

Answer:
"""
        )

        try:
            response = self.llm.invoke(
                answer_prompt.format(question=question, results=result_text, row_count=row_count))
            return response.strip()

        except Exception as e:
            print(f"Error generating answer: {e}")
            return "I apologize, but I encountered an error while generating the answer."

    def ask_question(self, question):
        """Main method to process a question and return an answer"""
        print(f"\n🤔 Processing question: {question}")

        try:
            # Step 1: Generate SQL query
            print("🔄 Generating SQL query...")
            sql_query = self.generate_sql_from_question(question)

            if not sql_query:
                return "I'm sorry, I couldn't generate a SQL query for your question."

            print(f"📝 Generated SQL: {sql_query}")

            # Step 2: Execute SQL query
            print("🔄 Executing query...")
            sql_result = self.execute_sql_query(sql_query)

            if sql_result is None:
                return "I encountered an error while executing the database query."

            # Step 3: Generate natural language answer
            print("🔄 Generating answer...")
            answer = self.generate_answer(question, sql_result)

            print("✅ Answer generated!")

            return {
                'answer': answer,
                'sql_query': sql_query,
                'raw_results': sql_result
            }

        except Exception as e:
            print(f"❌ Error processing question: {e}")
            return f"I apologize, but I encountered an error: {str(e)}"


# Test the system
if __name__ == "__main__":
    print("🚀 Initializing Donor RAG System with Real Data...")

    try:
        rag = DonorRAGSystem()

        # Test questions for real data
        test_questions = [
            "Who are our top 5 donors by lifetime value?",
            "Show me total donations by channel",
            "What's the total amount raised?",
            "Which donors have made more than 5 donations?",
            "Show me recent donations",
            "What age groups donate the most?",
            "Compare Tax vs Non-Tax donations"
        ]

        print("\n🧪 Testing with sample questions...")
        for question in test_questions:
            result = rag.ask_question(question)
            if isinstance(result, dict):
                print(f"\nQ: {question}")
                print(f"A: {result['answer']}")
                print(f"SQL: {result['sql_query']}")
                if not result['raw_results'].empty:
                    print(f"Results: {len(result['raw_results'])} rows returned")
                else:
                    print("Results: No data returned")
            else:
                print(f"\nQ: {question}")
                print(f"A: {result}")
            print("-" * 50)

    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        print("Make sure PostgreSQL is running and Ollama is available.")