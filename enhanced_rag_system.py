#!/usr/bin/env python3
"""
Enhanced Donor RAG System with Data Validation and Accuracy Checks
FIXED VERSION - PostgreSQL Compatible
Replace your enhanced_rag_system.py with this complete file
"""

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
import json

warnings.filterwarnings("ignore", category=UserWarning)

# Load environment variables
load_dotenv()


class DataValidationSystem:
    """System to validate data accuracy and query results"""

    def __init__(self, rag_system):
        self.rag_system = rag_system
        self.validation_cache = {}

    def validate_database_integrity(self):
        """Comprehensive database validation"""
        print("🔍 Running Database Integrity Check...")

        validation_results = {
            'total_donors': 0,
            'total_donations': 0,
            'total_amount': 0,
            'date_range': None,
            'issues': []
        }

        try:
            # Basic counts
            donor_count = self.rag_system.execute_sql_query("SELECT COUNT(DISTINCT donor_id) FROM donors;")
            donation_count = self.rag_system.execute_sql_query("SELECT COUNT(*) FROM donations;")
            total_amount = self.rag_system.execute_sql_query(
                "SELECT SUM(amount) FROM donations WHERE amount IS NOT NULL;")

            # Date range
            date_range = self.rag_system.execute_sql_query("""
                SELECT 
                    MIN(donation_date) as earliest_date,
                    MAX(donation_date) as latest_date,
                    COUNT(*) as donations_with_dates
                FROM donations 
                WHERE donation_date IS NOT NULL;
            """)

            # Store results
            if not donor_count.empty:
                validation_results['total_donors'] = int(donor_count.iloc[0]['count'])
            if not donation_count.empty:
                validation_results['total_donations'] = int(donation_count.iloc[0]['count'])
            if not total_amount.empty and total_amount.iloc[0]['sum'] is not None:
                validation_results['total_amount'] = float(total_amount.iloc[0]['sum'])
            if not date_range.empty:
                validation_results['date_range'] = {
                    'earliest': str(date_range.iloc[0]['earliest_date']),
                    'latest': str(date_range.iloc[0]['latest_date']),
                    'donations_with_dates': int(date_range.iloc[0]['donations_with_dates'])
                }

            # Cross-validation checks
            self._run_cross_validation_checks(validation_results)

            print("✅ Database Integrity Check Complete")
            return validation_results

        except Exception as e:
            print(f"❌ Database Validation Error: {e}")
            validation_results['issues'].append(f"Validation failed: {e}")
            return validation_results

    def _run_cross_validation_checks(self, results):
        """Run cross-validation checks"""
        try:
            # Check donor-donation consistency
            donor_donation_check = self.rag_system.execute_sql_query("""
                SELECT 
                    (SELECT COUNT(DISTINCT donor_id) FROM donations) as donors_in_donations,
                    (SELECT COUNT(*) FROM donors) as total_donors,
                    (SELECT COUNT(*) FROM donations WHERE donor_id NOT IN (SELECT donor_id FROM donors)) as orphaned_donations
            """)

            if not donor_donation_check.empty:
                row = donor_donation_check.iloc[0]
                if row['orphaned_donations'] > 0:
                    results['issues'].append(f"Found {row['orphaned_donations']} donations with invalid donor_ids")

                if row['donors_in_donations'] != row['total_donors']:
                    results['issues'].append(
                        f"Mismatch: {row['total_donors']} donors in donors table vs {row['donors_in_donations']} in donations")

            # Check for data quality issues
            data_quality = self.rag_system.execute_sql_query("""
                SELECT 
                    COUNT(*) FILTER (WHERE amount <= 0) as negative_amounts,
                    COUNT(*) FILTER (WHERE donation_date IS NULL) as missing_dates,
                    COUNT(*) FILTER (WHERE donor_id IS NULL) as missing_donor_ids
                FROM donations;
            """)

            if not data_quality.empty:
                row = data_quality.iloc[0]
                if row['negative_amounts'] > 0:
                    results['issues'].append(f"Found {row['negative_amounts']} donations with negative/zero amounts")
                if row['missing_dates'] > 0:
                    results['issues'].append(f"Found {row['missing_dates']} donations with missing dates")
                if row['missing_donor_ids'] > 0:
                    results['issues'].append(f"Found {row['missing_donor_ids']} donations with missing donor IDs")

        except Exception as e:
            results['issues'].append(f"Cross-validation error: {e}")

    def validate_query_result(self, question, sql_query, result_df):
        """Validate individual query results"""
        validation = {
            'is_valid': True,
            'warnings': [],
            'corrections': [],
            'confidence': 1.0
        }

        try:
            # Check for empty results on basic queries
            if any(word in question.lower() for word in ['total', 'count', 'how many']) and result_df.empty:
                validation['warnings'].append("Basic count query returned no results - check if data exists")
                validation['confidence'] *= 0.5

            # Validate donor vs donation queries
            if 'donor' in question.lower() and 'donation' not in question.lower():
                if 'COUNT(*)' in sql_query and 'DISTINCT donor_id' not in sql_query:
                    validation['warnings'].append("Query might be counting donations instead of unique donors")
                    validation['confidence'] *= 0.7

            # Check for reasonable amounts
            if not result_df.empty and any(col in result_df.columns for col in ['amount', 'total_amount', 'sum']):
                amount_cols = [col for col in result_df.columns if 'amount' in col.lower() or col == 'sum']
                for col in amount_cols:
                    if col in result_df.columns:
                        amounts = result_df[col].dropna()
                        if len(amounts) > 0:
                            max_amount = amounts.max()
                            if max_amount > 1000000:  # $1M+ donations are rare
                                validation['warnings'].append(f"Unusually high amount detected: ${max_amount:,.2f}")
                            if amounts.min() < 0:
                                validation['warnings'].append("Negative amounts detected")
                                validation['confidence'] *= 0.5

            # Validate time-based queries
            if any(word in question.lower() for word in ['year', 'month', 'quarter', '2023', '2024', '2025']):
                if result_df.empty:
                    validation['warnings'].append("Time-based query returned no results - check date filters")
                    validation['confidence'] *= 0.6

            return validation

        except Exception as e:
            validation['is_valid'] = False
            validation['warnings'].append(f"Validation error: {e}")
            return validation


class AccurateQueryGenerator:
    """Enhanced query generator with accuracy focus"""

    def __init__(self, rag_system):
        self.rag_system = rag_system

    def generate_validated_sql(self, question):
        """Generate SQL with built-in validation"""

        # Pre-processing: Clean and clarify the question
        cleaned_question = self._preprocess_question(question)

        # Determine query intent more precisely
        query_intent = self._analyze_query_intent(cleaned_question)

        # Generate SQL based on intent
        sql_query = self._generate_intent_based_sql(cleaned_question, query_intent)

        # Validate generated SQL
        validated_sql = self._validate_generated_sql(sql_query, query_intent)

        return validated_sql, query_intent

    def _preprocess_question(self, question):
        """Clean and standardize the question"""
        # Convert common phrases to standard forms
        replacements = {
            'how many donors': 'count distinct donors',
            'total donors': 'count distinct donors',
            'number of donors': 'count distinct donors',
            'donation amount': 'amount',
            'total raised': 'sum of amounts',
        }

        cleaned = question.lower()
        for old, new in replacements.items():
            cleaned = cleaned.replace(old, new)

        return cleaned

    def _analyze_query_intent(self, question):
        """Analyze what the user really wants - FIXED VERSION"""
        intent = {
            'metric_type': 'unknown',
            'entity_type': 'unknown',
            'time_filter': None,
            'aggregation': 'none'
        }

        # Determine if asking about donors or donations - FIXED ORDER
        if 'distinct donor' in question or 'unique donor' in question or 'count donor' in question:
            intent['entity_type'] = 'donors'
            intent['aggregation'] = 'count_distinct'
        elif 'average' in question or 'avg' in question:
            intent['entity_type'] = 'amounts'
            intent['aggregation'] = 'average'  # FIXED: was 'sum' before
        elif 'sum' in question or 'total raised' in question or 'total donation' in question:
            intent['entity_type'] = 'amounts'
            intent['aggregation'] = 'sum'
        elif 'count' in question:
            intent['entity_type'] = 'donations'
            intent['aggregation'] = 'count'

        # Time filters - FIXED: Added 2025
        time_patterns = {
            r'\b2023\b': "EXTRACT(YEAR FROM donation_date) = 2023",
            r'\b2024\b': "EXTRACT(YEAR FROM donation_date) = 2024",
            r'\b2025\b': "EXTRACT(YEAR FROM donation_date) = 2025",
            r'last month': "donation_date >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month') AND donation_date < DATE_TRUNC('month', CURRENT_DATE)",
            r'this year': "EXTRACT(YEAR FROM donation_date) = EXTRACT(YEAR FROM CURRENT_DATE)",
            r'q1': "EXTRACT(QUARTER FROM donation_date) = 1",
            r'q2': "EXTRACT(QUARTER FROM donation_date) = 2"
        }

        for pattern, sql in time_patterns.items():
            if re.search(pattern, question, re.IGNORECASE):
                intent['time_filter'] = sql
                break

        return intent

    def _generate_intent_based_sql(self, question, intent):
        """Generate SQL based on clear intent"""

        base_conditions = ["donation_date IS NOT NULL"]
        if intent['time_filter']:
            base_conditions.append(intent['time_filter'])

        where_clause = " AND ".join(base_conditions)

        # Generate SQL based on intent
        if intent['entity_type'] == 'donors' and intent['aggregation'] == 'count_distinct':
            sql = f"SELECT COUNT(DISTINCT donor_id) as donor_count FROM donations WHERE {where_clause};"

        elif intent['entity_type'] == 'amounts' and intent['aggregation'] == 'sum':
            sql = f"SELECT SUM(amount) as total_amount FROM donations WHERE {where_clause};"

        elif intent['entity_type'] == 'amounts' and intent['aggregation'] == 'average':
            sql = f"SELECT AVG(amount) as average_amount FROM donations WHERE {where_clause};"

        elif intent['entity_type'] == 'donations' and intent['aggregation'] == 'count':
            sql = f"SELECT COUNT(*) as donation_count FROM donations WHERE {where_clause};"

        elif 'top' in question and 'donor' in question:
            limit = self._extract_number(question) or 5
            sql = f"""
            SELECT d.donor_name, d.lifetime_value 
            FROM donors d 
            ORDER BY d.lifetime_value DESC 
            LIMIT {limit};
            """

        elif 'segment' in question.lower():
            sql = self._generate_segmentation_sql()

        else:
            # Fallback to enhanced prompt-based generation
            sql = self._generate_with_enhanced_prompt(question, intent)

        return sql

    def _generate_segmentation_sql(self):
        """Generate donor segmentation SQL - FIXED PostgreSQL VERSION"""
        return """
        WITH donor_segments AS (
            SELECT 
                d.donor_id,
                d.donor_name,
                d.lifetime_value,
                d.total_donations,
                d.age_range,
                CASE 
                    WHEN d.lifetime_value >= 10000 THEN 'Major Donors ($10K+)'
                    WHEN d.lifetime_value >= 5000 THEN 'Significant Donors ($5K-$10K)'
                    WHEN d.lifetime_value >= 1000 THEN 'Regular Donors ($1K-$5K)'
                    WHEN d.total_donations >= 5 THEN 'Frequent Small Donors'
                    WHEN d.last_donation_date IS NOT NULL AND (CURRENT_DATE - d.last_donation_date) > INTERVAL '365 days' THEN 'Lapsed Donors'
                    ELSE 'New/Occasional Donors'
                END as segment,
                CASE 
                    WHEN d.lifetime_value >= 10000 THEN 'VIP stewardship, personal attention, exclusive events'
                    WHEN d.lifetime_value >= 5000 THEN 'Regular updates, mid-level recognition, targeted appeals'
                    WHEN d.lifetime_value >= 1000 THEN 'Quarterly newsletters, annual thank you, upgrade campaigns'
                    WHEN d.total_donations >= 5 THEN 'Monthly emails, loyalty programs, small gift appreciation'
                    WHEN d.last_donation_date IS NOT NULL AND (CURRENT_DATE - d.last_donation_date) > INTERVAL '365 days' THEN 'Re-engagement campaigns, win-back offers'
                    ELSE 'Welcome series, education about impact, gentle cultivation'
                END as engagement_strategy
            FROM donors d
        )
        SELECT 
            segment,
            COUNT(*) as donor_count,
            SUM(lifetime_value) as total_value,
            AVG(lifetime_value) as avg_value,
            STRING_AGG(DISTINCT engagement_strategy, ' | ') as recommended_strategies
        FROM donor_segments
        GROUP BY segment
        ORDER BY SUM(lifetime_value) DESC;
        """

    def _extract_number(self, text):
        """Extract number from text (e.g., 'top 5' -> 5)"""
        numbers = re.findall(r'\b(\d+)\b', text)
        return int(numbers[0]) if numbers else None

    def _generate_with_enhanced_prompt(self, question, intent):
        """Fallback to LLM with enhanced prompt"""
        enhanced_prompt = f"""
        Generate a PostgreSQL query for: {question}

        Intent Analysis: {intent}

        CRITICAL RULES:
        1. Column name is 'amount' NOT 'donation_amount'
        2. Always use COUNT(DISTINCT donor_id) for donor counts
        3. Always add WHERE donation_date IS NOT NULL
        4. Use proper PostgreSQL date functions
        5. For donor queries, focus on donors table or use DISTINCT in donations table
        6. For date calculations use INTERVAL syntax: (CURRENT_DATE - date_column) > INTERVAL '365 days'

        Generate ONLY the SQL query:
        """

        try:
            response = self.rag_system.llm.invoke(enhanced_prompt)
            return self._clean_sql_response(response)
        except Exception as e:
            print(f"Error in enhanced prompt generation: {e}")
            return "SELECT COUNT(*) as error FROM donations;"

    def _clean_sql_response(self, response):
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

    def _validate_generated_sql(self, sql_query, intent):
        """Validate the generated SQL - FIXED VERSION"""

        # Check for common issues
        if intent['entity_type'] == 'donors' and 'DISTINCT donor_id' not in sql_query:
            if 'COUNT(' in sql_query and 'donors' not in sql_query:
                # Fix: Add DISTINCT for donor counts
                sql_query = sql_query.replace('COUNT(*)', 'COUNT(DISTINCT donor_id)')

        # FIXED: Only add WHERE clause for simple queries, not complex CTEs
        if ('WHERE' not in sql_query and
                'donation' in sql_query and
                'WITH' not in sql_query and  # Don't mess with CTEs
                'FROM donors' not in sql_query):  # Don't mess with donor table queries

            # Add basic WHERE clause for donations table queries only
            if 'ORDER BY' in sql_query:
                sql_query = sql_query.replace('ORDER BY', 'WHERE donation_date IS NOT NULL ORDER BY')
            elif 'GROUP BY' in sql_query:
                sql_query = sql_query.replace('GROUP BY', 'WHERE donation_date IS NOT NULL GROUP BY')
            else:
                sql_query = sql_query.rstrip(';') + ' WHERE donation_date IS NOT NULL;'

        return sql_query


class EnhancedDonorRAGSystem:
    """Enhanced RAG system with accuracy validation - FIXED VERSION"""

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

        # Initialize validation and accuracy components
        self.validator = DataValidationSystem(self)
        self.accurate_generator = AccurateQueryGenerator(self)

        # Run initial data validation
        self.database_stats = self.validator.validate_database_integrity()
        print(
            f"📊 Database Stats: {self.database_stats['total_donors']} donors, {self.database_stats['total_donations']} donations")

        if self.database_stats['issues']:
            print("⚠️ Data Issues Found:")
            for issue in self.database_stats['issues']:
                print(f"   - {issue}")

        print("✅ Enhanced RAG System with Validation initialized successfully!")

    def execute_sql_query(self, query):
        """Execute SQL query with enhanced error handling"""
        try:
            conn = psycopg2.connect(**self.db_params)
            df = pd.read_sql(query, conn)
            conn.close()
            return df
        except Exception as e:
            print(f"SQL Error: {e}")
            print(f"Query: {query}")
            return pd.DataFrame()

    def ask_question(self, question):
        """Enhanced question processing with validation"""
        print(f"\n🤔 Processing question with validation: {question}")

        try:
            # Generate validated SQL
            sql_query, query_intent = self.accurate_generator.generate_validated_sql(question)
            print(f"🔍 Query intent: {query_intent}")
            print(f"📝 Generated SQL: {sql_query}")

            # Execute query
            result_df = self.execute_sql_query(sql_query)

            # Validate results
            validation = self.validator.validate_query_result(question, sql_query, result_df)

            # Generate answer with validation context
            answer = self._generate_validated_answer(question, sql_query, result_df, validation, query_intent)

            return {
                'answer': answer,
                'sql_query': sql_query,
                'raw_results': result_df,
                'query_intent': query_intent,
                'validation': validation,
                'database_stats': self.database_stats
            }

        except Exception as e:
            print(f"❌ Error processing question: {e}")
            return f"I apologize, but I encountered an error: {str(e)}"

    def _generate_validated_answer(self, question, sql_query, result_df, validation, query_intent):
        """Generate answer with validation insights"""

        # Base answer generation
        if not result_df.empty:
            result_text = result_df.to_string(index=False)

            # Add validation warnings if present
            validation_notes = ""
            if validation['warnings']:
                validation_notes = f"\n\n⚠️ Data Quality Notes:\n" + "\n".join(
                    [f"• {w}" for w in validation['warnings']])

            # Add confidence indicator
            confidence_text = ""
            if validation['confidence'] < 0.8:
                confidence_text = f"\n\n📊 Confidence Level: {validation['confidence']:.0%} - Please verify results"

            prompt = f"""
            Provide a comprehensive answer based on the validated donor data results.

            Question: {question}
            Query Intent: {query_intent}

            Results ({len(result_df)} rows):
            {result_text}

            Database Context:
            - Total Donors in System: {self.database_stats['total_donors']:,}
            - Total Donations in System: {self.database_stats['total_donations']:,}
            - Total Amount Raised: ${self.database_stats['total_amount']:,.2f}

            Instructions:
            1. Provide specific numbers and percentages
            2. Reference the database context for perspective
            3. Include actionable insights
            4. Format currency clearly
            5. If this is a segmentation query, provide specific engagement strategies
            6. Be accurate about what the data shows

            Answer:
            """

            try:
                response = self.llm.invoke(prompt)
                return response.strip() + validation_notes + confidence_text
            except Exception as e:
                return f"Results found but error generating response: {result_text[:500]}..."

        else:
            return f"No results found for your query. This might indicate:\n• The time period has no data\n• The criteria are too restrictive\n• There may be a data issue\n\nDatabase contains {self.database_stats['total_donors']} donors and {self.database_stats['total_donations']} donations total."


# Test the enhanced system
if __name__ == "__main__":
    print("🚀 Initializing Enhanced RAG System with Data Validation...")

    try:
        rag = EnhancedDonorRAGSystem()

        # Test accuracy on the problematic queries
        test_questions = [
            "How many donors do we have?",
            "What was our average donation amount?",
            "Show me total donations for 2024",
            "List our top 5 donors by total contribution",
            "Segment our donor base by giving potential and suggest personalized engagement strategies for each segment"
        ]

        print("\n🧪 Testing enhanced accuracy...")
        for question in test_questions:
            result = rag.ask_question(question)
            if isinstance(result, dict):
                print(f"\nQ: {question}")
                print(f"Intent: {result['query_intent']}")
                print(f"Validation: {result['validation']['confidence']:.0%} confidence")
                if result['validation']['warnings']:
                    print(f"Warnings: {', '.join(result['validation']['warnings'])}")
                print(f"A: {result['answer'][:300]}...")
                print(f"Rows returned: {len(result['raw_results'])}")
            else:
                print(f"\nQ: {question}")
                print(f"A: {result}")
            print("-" * 80)

    except Exception as e:
        print(f"❌ Failed to initialize enhanced system: {e}")
        import traceback

        traceback.print_exc()