import os
import psycopg2
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from datetime import datetime, timedelta, date
import re
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Load environment variables
load_dotenv()


class TimeExpressionParser:
    """Parse natural language time expressions into SQL"""

    def __init__(self):
        self.current_year = datetime.now().year
        self.current_date = datetime.now().date()

    def parse_time_filter(self, text):
        """Convert time expressions to SQL WHERE clauses"""
        text = text.lower()

        # Year patterns
        year_match = re.search(r'\b(20\d{2})\b', text)
        if year_match:
            year = year_match.group(1)
            return f"EXTRACT(YEAR FROM donation_date) = {year}"

        # Quarter patterns
        if 'q1' in text or 'quarter 1' in text or 'first quarter' in text:
            year = self._extract_year_context(text)
            return f"EXTRACT(QUARTER FROM donation_date) = 1 AND EXTRACT(YEAR FROM donation_date) = {year}"
        elif 'q2' in text or 'quarter 2' in text or 'second quarter' in text:
            year = self._extract_year_context(text)
            return f"EXTRACT(QUARTER FROM donation_date) = 2 AND EXTRACT(YEAR FROM donation_date) = {year}"
        elif 'q3' in text or 'quarter 3' in text or 'third quarter' in text:
            year = self._extract_year_context(text)
            return f"EXTRACT(QUARTER FROM donation_date) = 3 AND EXTRACT(YEAR FROM donation_date) = {year}"
        elif 'q4' in text or 'quarter 4' in text or 'fourth quarter' in text:
            year = self._extract_year_context(text)
            return f"EXTRACT(QUARTER FROM donation_date) = 4 AND EXTRACT(YEAR FROM donation_date) = {year}"

        # Month patterns
        if 'last month' in text:
            return "donation_date >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month') AND donation_date < DATE_TRUNC('month', CURRENT_DATE)"
        elif 'this month' in text:
            return "donation_date >= DATE_TRUNC('month', CURRENT_DATE)"
        elif 'last 3 months' in text:
            return "donation_date >= CURRENT_DATE - INTERVAL '3 months'"
        elif 'last 6 months' in text:
            return "donation_date >= CURRENT_DATE - INTERVAL '6 months'"

        # Year patterns
        if 'this year' in text:
            return f"EXTRACT(YEAR FROM donation_date) = {self.current_year}"
        elif 'last year' in text:
            return f"EXTRACT(YEAR FROM donation_date) = {self.current_year - 1}"
        elif 'past 3 years' in text or 'last 3 years' in text:
            return f"donation_date >= CURRENT_DATE - INTERVAL '3 years'"

        return None

    def _extract_year_context(self, text):
        """Extract year from context, default to current year"""
        year_match = re.search(r'\b(20\d{2})\b', text)
        return year_match.group(1) if year_match else self.current_year


class AdvancedAnalytics:
    """Advanced analytics functions for donor data"""

    def __init__(self, rag_system):
        self.rag_system = rag_system

    def calculate_retention_rate(self, time_period="yearly"):
        """Calculate donor retention rates"""
        sql = """
        WITH yearly_donors AS (
            SELECT 
                donor_id,
                EXTRACT(YEAR FROM donation_date) as year
            FROM donations
            WHERE donation_date IS NOT NULL
            GROUP BY donor_id, EXTRACT(YEAR FROM donation_date)
        ),
        retention_analysis AS (
            SELECT 
                y1.year,
                COUNT(DISTINCT y1.donor_id) as total_donors,
                COUNT(DISTINCT y2.donor_id) as retained_donors
            FROM yearly_donors y1
            LEFT JOIN yearly_donors y2 ON y1.donor_id = y2.donor_id 
                AND y2.year = y1.year + 1
            GROUP BY y1.year
        )
        SELECT 
            year,
            total_donors,
            COALESCE(retained_donors, 0) as retained_donors,
            CASE 
                WHEN total_donors > 0 THEN ROUND((COALESCE(retained_donors, 0)::DECIMAL / total_donors * 100), 2)
                ELSE 0 
            END as retention_rate_percent
        FROM retention_analysis
        WHERE year < EXTRACT(YEAR FROM CURRENT_DATE)
        ORDER BY year;
        """
        return self.rag_system.execute_sql_query(sql)

    def donor_segmentation_rfm(self):
        """RFM Analysis - Fixed for PostgreSQL"""
        sql = """
        WITH donor_rfm AS (
            SELECT 
                d.donor_id,
                d.donor_name,
                d.age_range,
                (CURRENT_DATE - MAX(dn.donation_date))::INTEGER as recency_days,
                COUNT(dn.donation_id) as frequency,
                SUM(dn.amount) as monetary
            FROM donors d
            JOIN donations dn ON d.donor_id = dn.donor_id
            WHERE dn.donation_date IS NOT NULL
            GROUP BY d.donor_id, d.donor_name, d.age_range
        ),
        rfm_scores AS (
            SELECT *,
                CASE 
                    WHEN recency_days <= 90 THEN 5
                    WHEN recency_days <= 180 THEN 4
                    WHEN recency_days <= 365 THEN 3
                    WHEN recency_days <= 730 THEN 2
                    ELSE 1
                END as recency_score,
                CASE 
                    WHEN frequency >= 20 THEN 5
                    WHEN frequency >= 10 THEN 4
                    WHEN frequency >= 5 THEN 3
                    WHEN frequency >= 2 THEN 2
                    ELSE 1
                END as frequency_score,
                CASE 
                    WHEN monetary >= 10000 THEN 5
                    WHEN monetary >= 5000 THEN 4
                    WHEN monetary >= 1000 THEN 3
                    WHEN monetary >= 500 THEN 2
                    ELSE 1
                END as monetary_score
            FROM donor_rfm
        )
        SELECT *,
            CASE 
                WHEN recency_score >= 4 AND frequency_score >= 4 AND monetary_score >= 4 THEN 'Champions'
                WHEN recency_score >= 4 AND frequency_score >= 3 AND monetary_score >= 3 THEN 'Loyal Customers'
                WHEN recency_score >= 4 AND frequency_score <= 2 AND monetary_score >= 4 THEN 'New Big Spenders'
                WHEN recency_score >= 3 AND frequency_score >= 3 AND monetary_score >= 3 THEN 'Potential Loyalists'
                WHEN recency_score >= 4 AND frequency_score <= 2 AND monetary_score <= 2 THEN 'New Customers'
                WHEN recency_score <= 2 AND frequency_score >= 3 AND monetary_score >= 3 THEN 'At Risk'
                WHEN recency_score <= 2 AND frequency_score >= 4 AND monetary_score >= 4 THEN 'Cannot Lose Them'
                WHEN recency_score <= 1 AND frequency_score >= 2 AND monetary_score >= 2 THEN 'Hibernating'
                ELSE 'Lost'
            END as donor_segment
        FROM rfm_scores
        ORDER BY monetary DESC;
        """
        return self.rag_system.execute_sql_query(sql)

    def trend_analysis(self, metric="amount", period="monthly", years=3):
        """Analyze trends over time"""
        if period == "monthly":
            date_part = "TO_CHAR(donation_date, 'YYYY-MM')"
            date_label = "month"
        elif period == "quarterly":
            date_part = "CONCAT(EXTRACT(YEAR FROM donation_date)::TEXT, '-Q', EXTRACT(QUARTER FROM donation_date)::TEXT)"
            date_label = "quarter"
        else:  # yearly
            date_part = "EXTRACT(YEAR FROM donation_date)::TEXT"
            date_label = "year"

        if metric == "amount":
            metric_calc = "SUM(amount)"
            metric_label = "total_amount"
        elif metric == "count":
            metric_calc = "COUNT(*)"
            metric_label = "donation_count"
        else:  # average
            metric_calc = "AVG(amount)"
            metric_label = "avg_amount"

        sql = f"""
        SELECT 
            {date_part} as {date_label},
            {metric_calc} as {metric_label},
            COUNT(DISTINCT donor_id) as unique_donors
        FROM donations
        WHERE donation_date >= CURRENT_DATE - INTERVAL '{years} years'
            AND donation_date IS NOT NULL
            AND amount > 0
        GROUP BY {date_part}
        ORDER BY {date_part};
        """
        return self.rag_system.execute_sql_query(sql)

    def donor_acquisition_analysis(self):
        """Analyze new vs returning donors by period"""
        sql = """
        WITH first_donations AS (
            SELECT donor_id, MIN(donation_date) as first_donation_date
            FROM donations
            WHERE donation_date IS NOT NULL
            GROUP BY donor_id
        ),
        monthly_analysis AS (
            SELECT 
                TO_CHAR(d.donation_date, 'YYYY-MM') as month,
                COUNT(DISTINCT d.donor_id) as total_donors,
                COUNT(DISTINCT CASE WHEN DATE_TRUNC('month', fd.first_donation_date) = DATE_TRUNC('month', d.donation_date) THEN d.donor_id END) as new_donors,
                SUM(d.amount) as total_amount
            FROM donations d
            JOIN first_donations fd ON d.donor_id = fd.donor_id
            WHERE d.donation_date >= CURRENT_DATE - INTERVAL '2 years'
                AND d.donation_date IS NOT NULL
            GROUP BY TO_CHAR(d.donation_date, 'YYYY-MM')
        )
        SELECT 
            month,
            total_donors,
            new_donors,
            total_donors - new_donors as returning_donors,
            CASE 
                WHEN total_donors > 0 THEN ROUND((new_donors::DECIMAL / total_donors * 100), 2)
                ELSE 0 
            END as new_donor_percentage,
            total_amount
        FROM monthly_analysis
        ORDER BY month;
        """
        return self.rag_system.execute_sql_query(sql)


class EnhancedDonorRAGSystem:
    """Enhanced RAG system with intermediate analytics capabilities"""

    def __init__(self):
        """Initialize the enhanced RAG system"""
        self.db_params = {
            'host': os.getenv('DB_HOST'),
            'port': os.getenv('DB_PORT'),
            'database': os.getenv('DB_NAME'),
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD')
        }

        # Initialize components
        print("🔄 Loading sentence transformer model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        print("🔄 Connecting to Ollama...")
        try:
            self.llm = Ollama(
                model=os.getenv('OLLAMA_MODEL', 'mistral:7b'),
                base_url=os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
            )
            self.llm.invoke("test")
            print("✅ Ollama connection successful!")
        except Exception as e:
            print(f"❌ Ollama connection failed: {e}")
            raise

        # Initialize enhanced components
        self.time_parser = TimeExpressionParser()
        self.analytics = AdvancedAnalytics(self)

        # Initialize FAISS vector store
        self.vector_store = None
        self.documents = []

        # Create enhanced embeddings
        self.create_enhanced_vector_store()

        print("✅ Enhanced RAG System initialized successfully!")

    def create_enhanced_vector_store(self):
        """Create enhanced vector embeddings with analytical patterns"""
        print("🔄 Creating enhanced vector embeddings...")

        try:
            # Get schema and sample data
            schema = self.get_database_schema()
            sample_data = self.get_sample_data()

            documents = []

            # Add enhanced schema
            documents.append(f"Enhanced Database Schema:\n{schema}")

            # Add sample data with proper checking
            if sample_data and isinstance(sample_data, dict):
                if 'donors' in sample_data and sample_data['donors'] is not None and not sample_data['donors'].empty:
                    documents.append(f"Sample Donors:\n{sample_data['donors'].to_string(index=False)}")

                if 'donations' in sample_data and sample_data['donations'] is not None and not sample_data[
                    'donations'].empty:
                    documents.append(f"Sample Donations:\n{sample_data['donations'].to_string(index=False)}")

            # Add enhanced analytical patterns
            analytical_patterns = """
            ENHANCED SQL PATTERNS FOR INTERMEDIATE ANALYSIS:

            CRITICAL: Column name is 'amount' NOT 'donation_amount'

            1. Simple Time-based Queries:
            - "donations in 2024": SELECT SUM(amount), COUNT(*) FROM donations WHERE EXTRACT(YEAR FROM donation_date) = 2024;
            - "total for this year": SELECT SUM(amount) as total FROM donations WHERE EXTRACT(YEAR FROM donation_date) = EXTRACT(YEAR FROM CURRENT_DATE);

            2. Quarterly Comparison (CORRECT):
            WITH q1_data AS (
                SELECT SUM(amount) as total_amount, COUNT(*) as donation_count
                FROM donations 
                WHERE EXTRACT(QUARTER FROM donation_date) = 1 AND EXTRACT(YEAR FROM donation_date) = 2024
            ),
            q2_data AS (
                SELECT SUM(amount) as total_amount, COUNT(*) as donation_count
                FROM donations 
                WHERE EXTRACT(QUARTER FROM donation_date) = 2 AND EXTRACT(YEAR FROM donation_date) = 2024
            )
            SELECT 'Q1' as period, q1.total_amount, q1.donation_count FROM q1_data q1
            UNION ALL
            SELECT 'Q2' as period, q2.total_amount, q2.donation_count FROM q2_data q2;

            3. Simple Aggregations:
            - Total donations: SELECT SUM(amount) FROM donations WHERE donation_date IS NOT NULL;
            - Count by year: SELECT EXTRACT(YEAR FROM donation_date) as year, COUNT(*), SUM(amount) FROM donations GROUP BY year;

            RULES:
            - Column is 'amount' NOT 'donation_amount'
            - Always use WHERE donation_date IS NOT NULL
            - Keep queries simple for basic questions
            - Use CTEs for comparisons
            """
            documents.append(analytical_patterns)

            self.documents = documents

            # Create embeddings
            embeddings = self.embedding_model.encode(documents)

            # Create FAISS index
            dimension = embeddings.shape[1]
            self.vector_store = faiss.IndexFlatL2(dimension)
            self.vector_store.add(embeddings.astype('float32'))

            print(f"✅ Enhanced vector store created with {len(documents)} documents")

        except Exception as e:
            print(f"❌ Error creating enhanced vector store: {e}")
            self.documents = []

    def get_database_schema(self):
        """Enhanced database schema information"""
        return """
        Enhanced Database Schema for Advanced Analytics:

        Table: donors
        - donor_id, donor_name: Unique identifiers
        - age_range: Demographics ('35-45', 'Over 56', etc.)
        - donor_type: Individual, Corporation, etc.
        - lifetime_value: Pre-calculated total giving
        - first_donation_date, last_donation_date: Giving timeline

        Table: donations
        - donation_id, donor_id: Identifiers
        - amount: Donation amount (CRITICAL: use 'amount' not 'donation_amount')
        - donation_date: Date field (use EXTRACT, DATE_TRUNC for time analysis)
        - channel: Payment method (Giro, Cash, Cheque, Bank Transfer, etc.)
        - donation_type: Tax classification

        Key Fields for Analysis:
        - Time: donation_date (supports YEAR, QUARTER, MONTH extraction)
        - Amount: amount (supports SUM, AVG, growth calculations)
        - Segmentation: age_range, channel, donor_type
        - Frequency: COUNT(*) GROUP BY donor_id
        """

    def get_sample_data(self):
        """Get enhanced sample data with proper error handling"""
        try:
            conn = psycopg2.connect(**self.db_params)

            # Enhanced donor sample
            donors_df = pd.read_sql("""
            SELECT donor_name, age_range, donor_type, total_donations, lifetime_value,
                   first_donation_date, last_donation_date
            FROM donors ORDER BY lifetime_value DESC LIMIT 3
            """, conn)

            # Enhanced donations sample
            donations_df = pd.read_sql("""
            SELECT donor_id, amount, donation_date, channel, donation_type
            FROM donations 
            WHERE donation_date IS NOT NULL
            ORDER BY donation_date DESC LIMIT 3
            """, conn)

            conn.close()
            return {'donors': donors_df, 'donations': donations_df}

        except Exception as e:
            print(f"Error getting sample data: {e}")
            return {}

    def classify_query_type(self, question):
        """Classify the type of analytical query"""
        question_lower = question.lower()

        if any(word in question_lower for word in ['retention', 'retain', 'return', 'come back']):
            return 'retention_analysis'
        elif any(word in question_lower for word in ['segment', 'group', 'category', 'rfm', 'behavior']):
            return 'segmentation'
        elif any(word in question_lower for word in ['trend', 'compare', 'vs', 'versus', 'growth', 'change']):
            return 'trend_analysis'
        elif any(word in question_lower for word in ['new donor', 'acquisition', 'first time']):
            return 'acquisition_analysis'
        elif self.time_parser.parse_time_filter(question):
            return 'time_based'
        else:
            return 'standard'

    def generate_trend_analysis_sql(self, question):
        """Generate specific SQL for trend analysis"""
        question_lower = question.lower()

        # Q1 vs Q2 comparison
        if 'q1' in question_lower and 'q2' in question_lower:
            year = 2024  # Default to current year
            year_match = re.search(r'\b(20\d{2})\b', question_lower)
            if year_match:
                year = year_match.group(1)

            sql = f"""
            WITH q1_data AS (
                SELECT 
                    SUM(amount) as total_amount,
                    COUNT(*) as donation_count,
                    COUNT(DISTINCT donor_id) as unique_donors
                FROM donations 
                WHERE EXTRACT(QUARTER FROM donation_date) = 1 
                    AND EXTRACT(YEAR FROM donation_date) = {year}
                    AND donation_date IS NOT NULL
            ),
            q2_data AS (
                SELECT 
                    SUM(amount) as total_amount,
                    COUNT(*) as donation_count,
                    COUNT(DISTINCT donor_id) as unique_donors
                FROM donations 
                WHERE EXTRACT(QUARTER FROM donation_date) = 2 
                    AND EXTRACT(YEAR FROM donation_date) = {year}
                    AND donation_date IS NOT NULL
            )
            SELECT 
                'Q1' as period, q1.total_amount, q1.donation_count, q1.unique_donors
            FROM q1_data q1
            UNION ALL
            SELECT 
                'Q2' as period, q2.total_amount, q2.donation_count, q2.unique_donors
            FROM q2_data q2
            ORDER BY period;
            """
            return sql

        # For other trend queries, return None to use analytics methods
        return None

    def generate_enhanced_sql(self, question):
        """Generate enhanced SQL for complex queries"""
        context = self.retrieve_context(question)
        time_filter = self.time_parser.parse_time_filter(question)

        enhanced_prompt = PromptTemplate(
            input_variables=["question", "context", "time_filter"],
            template="""
You are a PostgreSQL expert specializing in donor analytics. Generate a simple, correct SQL query.

Context and Patterns:
{context}

Question: {question}

Time Filter Detected: {time_filter}

CRITICAL RULES:
1. Column name is 'amount' NOT 'donation_amount'
2. If time_filter is provided, include it in WHERE clause
3. Always add WHERE donation_date IS NOT NULL
4. Keep queries simple - avoid overly complex CTEs unless necessary
5. For basic totals, use simple SELECT SUM(amount) FROM donations
6. Use proper PostgreSQL syntax

EXAMPLES:
- Simple total: SELECT SUM(amount) FROM donations WHERE donation_date IS NOT NULL;
- By year: SELECT EXTRACT(YEAR FROM donation_date) as year, SUM(amount) FROM donations WHERE donation_date IS NOT NULL GROUP BY year;

Generate ONE simple, working SQL query:
"""
        )

        try:
            response = self.llm.invoke(enhanced_prompt.format(
                question=question,
                context=context,
                time_filter=time_filter or "None detected"
            ))

            sql_query = self.clean_sql_response(response)

            # If time filter detected, ensure it's in the query
            if time_filter and time_filter not in sql_query:
                sql_query = self.inject_time_filter(sql_query, time_filter)

            return sql_query

        except Exception as e:
            print(f"Error generating enhanced SQL: {e}")
            return None

    def clean_sql_response(self, response):
        """Clean and format SQL response"""
        sql_query = response.strip()

        # Remove code blocks
        if "```sql" in sql_query:
            sql_query = sql_query.split("```sql")[1].split("```")[0]
        elif "```" in sql_query:
            sql_query = sql_query.split("```")[1]

        # Remove comments and explanations
        lines = sql_query.split('\n')
        clean_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('--') and not line.startswith('/*'):
                clean_lines.append(line)

        sql_query = ' '.join(clean_lines).strip()
        sql_query = sql_query.rstrip(';') + ';'

        return sql_query

    def inject_time_filter(self, sql_query, time_filter):
        """Inject time filter into existing SQL query"""
        if 'WHERE' in sql_query.upper():
            # Add to existing WHERE clause
            where_pos = sql_query.upper().find('WHERE')
            before_where = sql_query[:where_pos + 5]
            after_where = sql_query[where_pos + 5:]
            return f"{before_where} {time_filter} AND {after_where}"
        else:
            # Add new WHERE clause before GROUP BY or ORDER BY
            group_pos = sql_query.upper().find('GROUP BY')
            order_pos = sql_query.upper().find('ORDER BY')

            if group_pos != -1:
                insert_pos = group_pos
            elif order_pos != -1:
                insert_pos = order_pos
            else:
                insert_pos = len(sql_query) - 1  # Before semicolon

            before = sql_query[:insert_pos]
            after = sql_query[insert_pos:]
            return f"{before} WHERE {time_filter} {after}"

    def retrieve_context(self, query, k=3):
        """Enhanced context retrieval"""
        if not self.vector_store or not self.documents:
            return ""

        try:
            query_embedding = self.embedding_model.encode([query])
            distances, indices = self.vector_store.search(query_embedding.astype('float32'), k)

            relevant_docs = []
            for idx in indices[0]:
                if idx < len(self.documents):
                    relevant_docs.append(self.documents[idx])

            return "\n\n".join(relevant_docs)

        except Exception as e:
            print(f"Error retrieving context: {e}")
            return ""

    def execute_sql_query(self, query):
        """Execute SQL query with enhanced error handling"""
        try:
            conn = psycopg2.connect(**self.db_params)
            df = pd.read_sql(query, conn)
            conn.close()
            return df
        except Exception as e:
            print(f"SQL Error: {e}")
            return None

    def generate_enhanced_answer(self, question, sql_result, query_type=None):
        """Generate enhanced answers with analytical insights"""
        if sql_result is not None and not sql_result.empty:
            result_text = sql_result.to_string(index=False)
            row_count = len(sql_result)

            # Add analytical insights based on query type
            insights = self.generate_insights(sql_result, query_type)
        else:
            result_text = "No results found."
            row_count = 0
            insights = ""

        enhanced_prompt = PromptTemplate(
            input_variables=["question", "results", "row_count", "insights"],
            template="""
Provide a comprehensive analytical answer based on the donor data results.

Question: {question}
Number of results: {row_count}

Query Results:
{results}

Additional Insights:
{insights}

Instructions:
1. Provide specific numbers, percentages, and trends
2. Include comparative analysis when relevant
3. Highlight key insights and patterns
4. Format currency clearly (e.g., $1,234.56)
5. Mention growth rates, changes, or trends when applicable
6. Use professional but accessible language
7. Include actionable insights for fundraising strategy

Answer:
"""
        )

        try:
            response = self.llm.invoke(enhanced_prompt.format(
                question=question,
                results=result_text,
                row_count=row_count,
                insights=insights
            ))
            return response.strip()
        except Exception as e:
            print(f"Error generating enhanced answer: {e}")
            return "I apologize, but I encountered an error while generating the enhanced answer."

    def generate_insights(self, df, query_type):
        """Generate analytical insights from results"""
        if df.empty:
            return ""

        insights = []

        # Growth rate calculations
        if 'total_amount' in df.columns and len(df) > 1:
            amounts = df['total_amount'].tolist()
            if len(amounts) >= 2:
                growth_rate = ((amounts[-1] - amounts[0]) / amounts[0]) * 100
                insights.append(f"Growth rate: {growth_rate:.1f}% from first to last period")

        # Percentage distributions
        if 'donation_count' in df.columns:
            total_donations = df['donation_count'].sum()
            insights.append(f"Total donations analyzed: {total_donations:,}")

        # Top performer identification
        if len(df) > 1:
            top_row = df.iloc[0]
            insights.append(f"Top performer: {top_row.to_dict()}")

        return " | ".join(insights)

    def ask_question(self, question):
        """Enhanced question processing with intermediate analytics"""
        print(f"\n🤔 Processing enhanced question: {question}")

        try:
            query_type = self.classify_query_type(question)
            print(f"🔍 Query type detected: {query_type}")

            # Generate enhanced SQL or use pre-built analytics
            if query_type in ['retention_analysis', 'segmentation', 'acquisition_analysis']:
                print("🔄 Running specialized analytics...")
                if query_type == 'retention_analysis':
                    sql_result = self.analytics.calculate_retention_rate()
                    sql_query = "-- Donor Retention Analysis Query (Pre-built Analytics)"
                elif query_type == 'segmentation':
                    sql_result = self.analytics.donor_segmentation_rfm()
                    sql_query = "-- RFM Donor Segmentation Query (Pre-built Analytics)"
                elif query_type == 'acquisition_analysis':
                    sql_result = self.analytics.donor_acquisition_analysis()
                    sql_query = "-- Donor Acquisition Analysis Query (Pre-built Analytics)"
            elif query_type == 'trend_analysis':
                print("🔄 Running trend analysis...")
                # Try to get SQL string first
                sql_query = self.generate_trend_analysis_sql(question)
                if sql_query and isinstance(sql_query, str):
                    print(f"📝 Generated SQL: {sql_query}")
                    sql_result = self.execute_sql_query(sql_query)
                else:
                    # Fall back to analytics method
                    if 'monthly' in question.lower():
                        sql_result = self.analytics.trend_analysis(period='monthly')
                    elif 'quarterly' in question.lower():
                        sql_result = self.analytics.trend_analysis(period='quarterly')
                    else:
                        sql_result = self.analytics.trend_analysis(period='yearly')
                    sql_query = "-- Trend Analysis Query (Pre-built Analytics)"
            else:
                print("🔄 Generating enhanced SQL query...")
                sql_query = self.generate_enhanced_sql(question)

                if not sql_query:
                    return "I'm sorry, I couldn't generate a SQL query for your question."

                print(f"📝 Generated SQL: {sql_query}")

                print("🔄 Executing query...")
                sql_result = self.execute_sql_query(sql_query)

            if sql_result is None:
                return "I encountered an error while executing the database query."

            print("🔄 Generating enhanced answer...")
            answer = self.generate_enhanced_answer(question, sql_result, query_type)

            print("✅ Enhanced answer generated!")

            return {
                'answer': answer,
                'sql_query': sql_query,
                'raw_results': sql_result,
                'query_type': query_type,
                'insights': self.generate_insights(sql_result, query_type)
            }

        except Exception as e:
            print(f"❌ Error processing enhanced question: {e}")
            import traceback
            traceback.print_exc()
            return f"I apologize, but I encountered an error: {str(e)}"


# Test the enhanced system
if __name__ == "__main__":
    print("🚀 Initializing Final Fixed Enhanced Donor RAG System...")

    try:
        rag = EnhancedDonorRAGSystem()

        # Test intermediate complexity questions
        test_questions = [
            "Show me total donations for 2024",
            "Compare donation trends between Q1 and Q2",
            "Analyze donor retention rates over the past 3 years",
            "Show me donation patterns by demographic segment",
            "What's our donor acquisition trend this year?",
            "Segment our donors by giving behavior"
        ]

        print("\n🧪 Testing final fixed enhanced capabilities...")
        for question in test_questions:
            result = rag.ask_question(question)
            if isinstance(result, dict):
                print(f"\nQ: {question}")
                print(f"Type: {result.get('query_type', 'standard')}")
                print(f"A: {result['answer'][:200]}...")  # Truncate for readability
                if not result['raw_results'].empty:
                    print(f"Results: {len(result['raw_results'])} rows returned")
                else:
                    print("Results: No data returned")
            else:
                print(f"\nQ: {question}")
                print(f"A: {result}")
            print("-" * 50)

    except Exception as e:
        print(f"❌ Failed to initialize final fixed enhanced system: {e}")
        import traceback

        traceback.print_exc()
        print("Make sure PostgreSQL is running and Ollama is available.")