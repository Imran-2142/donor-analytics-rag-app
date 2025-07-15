import psycopg2
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()


def verify_database_accuracy():
    """Comprehensive database verification to identify data issues"""

    db_params = {
        'host': os.getenv('DB_HOST'),
        'port': os.getenv('DB_PORT'),
        'database': os.getenv('DB_NAME'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD')
    }

    print("🔍 COMPREHENSIVE DATABASE VERIFICATION")
    print("=" * 60)

    try:
        conn = psycopg2.connect(**db_params)

        # 1. Basic Table Counts
        print("\n📊 BASIC COUNTS:")
        print("-" * 30)

        tables_info = pd.read_sql("""
            SELECT 
                'donors' as table_name,
                COUNT(*) as row_count
            FROM donors
            UNION ALL
            SELECT 
                'donations' as table_name,
                COUNT(*) as row_count
            FROM donations;
        """, conn)

        for _, row in tables_info.iterrows():
            print(f"{row['table_name'].capitalize()}: {row['row_count']:,} records")

        # 2. Donor Analysis
        print("\n👥 DONOR ANALYSIS:")
        print("-" * 30)

        donor_analysis = pd.read_sql("""
            SELECT 
                COUNT(*) as total_donor_records,
                COUNT(DISTINCT donor_id) as unique_donor_ids,
                COUNT(DISTINCT donor_name) as unique_donor_names,
                MIN(lifetime_value) as min_lifetime_value,
                MAX(lifetime_value) as max_lifetime_value,
                AVG(lifetime_value) as avg_lifetime_value
            FROM donors;
        """, conn)

        row = donor_analysis.iloc[0]
        print(f"Total donor records: {row['total_donor_records']:,}")
        print(f"Unique donor IDs: {row['unique_donor_ids']:,}")
        print(f"Unique donor names: {row['unique_donor_names']:,}")
        print(f"Lifetime value range: ${row['min_lifetime_value']:,.2f} - ${row['max_lifetime_value']:,.2f}")
        print(f"Average lifetime value: ${row['avg_lifetime_value']:,.2f}")

        # 3. Donation Analysis
        print("\n💰 DONATION ANALYSIS:")
        print("-" * 30)

        donation_analysis = pd.read_sql("""
            SELECT 
                COUNT(*) as total_donation_records,
                COUNT(DISTINCT donor_id) as unique_donors_in_donations,
                COUNT(*) FILTER (WHERE amount IS NULL OR amount <= 0) as invalid_amounts,
                COUNT(*) FILTER (WHERE donation_date IS NULL) as missing_dates,
                MIN(amount) as min_amount,
                MAX(amount) as max_amount,
                AVG(amount) as avg_amount,
                SUM(amount) as total_amount,
                MIN(donation_date) as earliest_date,
                MAX(donation_date) as latest_date
            FROM donations;
        """, conn)

        row = donation_analysis.iloc[0]
        print(f"Total donation records: {row['total_donation_records']:,}")
        print(f"Unique donors with donations: {row['unique_donors_in_donations']:,}")
        print(f"Invalid amounts: {row['invalid_amounts']:,}")
        print(f"Missing dates: {row['missing_dates']:,}")
        print(f"Amount range: ${row['min_amount']:,.2f} - ${row['max_amount']:,.2f}")
        print(f"Average donation: ${row['avg_amount']:,.2f}")
        print(f"Total raised: ${row['total_amount']:,.2f}")
        print(f"Date range: {row['earliest_date']} to {row['latest_date']}")

        # 4. Test Queries
        print("\n🧪 TEST QUERY RESULTS:")
        print("-" * 30)

        test_queries = {
            "Total unique donors": "SELECT COUNT(DISTINCT donor_id) as count FROM donations;",
            "Total donation records": "SELECT COUNT(*) as count FROM donations;",
            "Total amount raised": "SELECT SUM(amount) as total FROM donations WHERE amount > 0;",
            "Average donation": "SELECT AVG(amount) as avg_amount FROM donations WHERE amount > 0;"
        }

        for description, query in test_queries.items():
            try:
                result = pd.read_sql(query, conn)
                if not result.empty:
                    value = result.iloc[0, 0]
                    if 'amount' in description.lower() or 'total' in description.lower():
                        print(f"{description}: ${value:,.2f}")
                    else:
                        print(f"{description}: {value:,}")
                else:
                    print(f"{description}: No result")
            except Exception as e:
                print(f"{description}: Error - {e}")

        conn.close()

        print("\n" + "=" * 60)
        print("✅ DATABASE VERIFICATION COMPLETE")

    except Exception as e:
        print(f"❌ Database verification failed: {e}")


if __name__ == "__main__":
    verify_database_accuracy()