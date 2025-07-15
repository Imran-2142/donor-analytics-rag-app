import psycopg2
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv
import numpy as np

load_dotenv()


def create_database():
    """Create the donor analytics database and tables"""

    # Database connection parameters
    db_params = {
        'host': os.getenv('DB_HOST'),
        'port': os.getenv('DB_PORT'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD')
    }

    try:
        # Connect to PostgreSQL server (not specific database)
        conn = psycopg2.connect(**db_params)
        conn.autocommit = True
        cursor = conn.cursor()

        # Create database if it doesn't exist
        cursor.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = 'donor_analytics'")
        exists = cursor.fetchone()
        if not exists:
            cursor.execute('CREATE DATABASE donor_analytics')
            print("✅ Database 'donor_analytics' created successfully!")
        else:
            print("✅ Database 'donor_analytics' already exists!")

        cursor.close()
        conn.close()

        # Connect to the new database
        db_params['database'] = 'donor_analytics'
        conn = psycopg2.connect(**db_params)
        conn.autocommit = False  # Use transactions
        cursor = conn.cursor()

        # Drop existing tables to start fresh
        cursor.execute('DROP TABLE IF EXISTS donations CASCADE')
        cursor.execute('DROP TABLE IF EXISTS donors CASCADE')
        conn.commit()

        # Create donors table
        cursor.execute('''
        CREATE TABLE donors (
            donor_id VARCHAR(50) PRIMARY KEY,
            donor_name VARCHAR(100) NOT NULL,
            age_range VARCHAR(50),
            donor_id_number VARCHAR(50),
            donor_id_type VARCHAR(20),
            donor_type VARCHAR(50),
            total_donations INTEGER DEFAULT 0,
            lifetime_value DECIMAL(12,2) DEFAULT 0,
            first_donation_date DATE,
            last_donation_date DATE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')

        # Create donations table
        cursor.execute('''
        CREATE TABLE donations (
            donation_id VARCHAR(50) PRIMARY KEY,
            donor_id VARCHAR(50) REFERENCES donors(donor_id),
            amount DECIMAL(10,2) NOT NULL,
            donation_date DATE,
            channel VARCHAR(50),
            donation_type VARCHAR(100),
            donor_tier_name VARCHAR(100),
            donor_tier_amount DECIMAL(10,2),
            donor_tier_description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')

        conn.commit()
        print("✅ Database tables created successfully!")

        # Load and insert real data
        success = load_csv_data(cursor, conn)

        cursor.close()
        conn.close()

        return success

    except Exception as e:
        print(f"❌ Error creating database: {e}")
        return False


def parse_date(date_str):
    """Parse various date formats"""
    if not date_str or pd.isna(date_str) or str(date_str).strip() == '':
        return None

    try:
        # Handle various date formats
        date_str = str(date_str).strip()
        if ' 0:00' in date_str:
            date_str = date_str.replace(' 0:00', '')

        # Try different date formats
        formats = ['%d/%m/%Y', '%m/%d/%Y', '%Y-%m-%d', '%d-%m-%Y']
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt).date()
            except ValueError:
                continue

        return None

    except Exception:
        return None


def load_csv_data(cursor, conn):
    """Load data from the CSV file with robust error handling"""

    csv_file = os.getenv('CSV_FILE', 'DonationData_anonymized.csv')

    try:
        # Read CSV file
        print(f"📁 Reading CSV file: {csv_file}")
        df = pd.read_csv(csv_file)

        print(f"📊 Loaded {len(df)} donation records")
        print(f"📊 Columns: {list(df.columns)}")

        # Clean column names
        df.columns = df.columns.str.strip()

        # Clean and validate data
        print("🧹 Cleaning and validating data...")

        # Remove rows without essential data
        df = df.dropna(subset=['Donor', 'Donation Amount'])
        df['Donation Amount'] = pd.to_numeric(df['Donation Amount'], errors='coerce')
        df = df[df['Donation Amount'] > 0]

        print(f"📊 After cleaning: {len(df)} valid donation records")

        # Parse dates
        print("📅 Parsing donation dates...")
        df['parsed_date'] = df['Donation Date'].apply(parse_date)

        # Create donor aggregations
        print("👥 Creating donor aggregations...")

        donor_stats = []

        for donor_name in df['Donor'].unique():
            if pd.isna(donor_name):
                continue

            donor_data = df[df['Donor'] == donor_name]

            # Get donor info
            age_range = donor_data['Age Range'].iloc[0] if not pd.isna(donor_data['Age Range'].iloc[0]) else None
            donor_id_number = donor_data['Donor ID Number'].iloc[0] if not pd.isna(
                donor_data['Donor ID Number'].iloc[0]) else None
            donor_id_type = donor_data['Donor ID Type'].iloc[0] if not pd.isna(
                donor_data['Donor ID Type'].iloc[0]) else None
            donor_type = donor_data['Donor Type'].iloc[0] if not pd.isna(donor_data['Donor Type'].iloc[0]) else None

            # Calculate aggregations
            total_donations = len(donor_data)
            lifetime_value = float(donor_data['Donation Amount'].sum())

            # Get date range
            valid_dates = donor_data['parsed_date'].dropna()
            first_date = valid_dates.min() if not valid_dates.empty else None
            last_date = valid_dates.max() if not valid_dates.empty else None

            donor_stats.append({
                'donor_id': str(donor_name),
                'donor_name': str(donor_name),
                'age_range': str(age_range) if age_range else None,
                'donor_id_number': str(donor_id_number) if donor_id_number else None,
                'donor_id_type': str(donor_id_type) if donor_id_type else None,
                'donor_type': str(donor_type) if donor_type else None,
                'total_donations': total_donations,
                'lifetime_value': lifetime_value,
                'first_donation_date': first_date,
                'last_donation_date': last_date
            })

        print(f"👥 Prepared {len(donor_stats)} donor records")

        # Insert donors with transaction handling
        print("💾 Inserting donor records...")
        donor_count = 0

        for donor in donor_stats:
            try:
                cursor.execute('''
                INSERT INTO donors (
                    donor_id, donor_name, age_range, donor_id_number, 
                    donor_id_type, donor_type, total_donations, lifetime_value,
                    first_donation_date, last_donation_date
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ''', (
                    donor['donor_id'],
                    donor['donor_name'],
                    donor['age_range'],
                    donor['donor_id_number'],
                    donor['donor_id_type'],
                    donor['donor_type'],
                    donor['total_donations'],
                    donor['lifetime_value'],
                    donor['first_donation_date'],
                    donor['last_donation_date']
                ))
                donor_count += 1

                # Commit every 100 records
                if donor_count % 100 == 0:
                    conn.commit()
                    print(f"   💾 Committed {donor_count} donors...")

            except Exception as e:
                print(f"⚠️  Error inserting donor {donor['donor_name']}: {e}")
                conn.rollback()  # Rollback this transaction

                # Try to continue with next donor
                try:
                    # Start new transaction
                    pass
                except:
                    break

        # Final commit for donors
        conn.commit()
        print(f"✅ Successfully inserted {donor_count} donor records")

        # Insert donations with batch processing
        print("💰 Inserting donation records...")

        donation_count = 0
        batch_size = 100

        # Process donations in batches
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i + batch_size]
            batch_success = 0

            try:
                for _, donation in batch_df.iterrows():
                    try:
                        # Validate required fields
                        if pd.isna(donation['Donor']) or pd.isna(donation['Donation Amount']):
                            continue

                        # Clean and prepare values
                        donation_id = str(donation['Donation ID']) if not pd.isna(
                            donation['Donation ID']) else f"DON_{i}_{donation_count}"
                        donor_id = str(donation['Donor'])
                        amount = float(donation['Donation Amount'])
                        donation_date = donation['parsed_date']
                        channel = str(donation['Channel']) if not pd.isna(donation['Channel']) else None
                        donation_type = str(donation['Donation Type']) if not pd.isna(
                            donation['Donation Type']) else None

                        # Handle tier information
                        tier_name = None
                        tier_amount = None
                        tier_desc = None

                        if 'Donor Tier/Name' in donation and not pd.isna(donation['Donor Tier/Name']):
                            tier_name = str(donation['Donor Tier/Name'])
                        if 'Donor Tier/Tier Amount' in donation and not pd.isna(donation['Donor Tier/Tier Amount']):
                            tier_amount = float(donation['Donor Tier/Tier Amount'])
                        if 'Donor Tier/Description' in donation and not pd.isna(donation['Donor Tier/Description']):
                            tier_desc = str(donation['Donor Tier/Description'])

                        cursor.execute('''
                        INSERT INTO donations (
                            donation_id, donor_id, amount, donation_date, channel,
                            donation_type, donor_tier_name, donor_tier_amount, donor_tier_description
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ''', (
                            donation_id, donor_id, amount, donation_date, channel,
                            donation_type, tier_name, tier_amount, tier_desc
                        ))

                        batch_success += 1
                        donation_count += 1

                    except Exception as e:
                        print(f"⚠️  Error inserting donation {donation.get('Donation ID', 'Unknown')}: {e}")
                        continue

                # Commit batch
                conn.commit()
                print(f"   💾 Committed batch {i // batch_size + 1}: {batch_success} donations")

            except Exception as e:
                print(f"⚠️  Batch error: {e}")
                conn.rollback()
                continue

        print(f"✅ Successfully inserted {donation_count} donation records")

        # Print final summary
        cursor.execute("SELECT COUNT(*) FROM donors")
        final_donor_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM donations")
        final_donation_count = cursor.fetchone()[0]

        cursor.execute("SELECT SUM(amount) FROM donations WHERE amount IS NOT NULL")
        total_amount = cursor.fetchone()[0] or 0

        cursor.execute("SELECT AVG(amount) FROM donations WHERE amount IS NOT NULL")
        avg_amount = cursor.fetchone()[0] or 0

        print(f"\n📈 Final Database Summary:")
        print(f"   👥 Total Donors: {final_donor_count}")
        print(f"   💰 Total Donations: {final_donation_count}")
        print(f"   💵 Total Amount: ${total_amount:,.2f}")
        print(f"   📊 Average Donation: ${avg_amount:.2f}")

        # Test queries
        if final_donation_count > 0:
            print(f"\n🧪 Testing database queries...")

            cursor.execute("SELECT channel, COUNT(*) FROM donations WHERE channel IS NOT NULL GROUP BY channel LIMIT 5")
            channels = cursor.fetchall()
            print(f"   📊 Top Channels: {channels}")

            cursor.execute(
                "SELECT age_range, COUNT(*) FROM donors WHERE age_range IS NOT NULL GROUP BY age_range LIMIT 5")
            ages = cursor.fetchall()
            print(f"   👥 Age Ranges: {ages}")

            cursor.execute("SELECT donor_name, lifetime_value FROM donors ORDER BY lifetime_value DESC LIMIT 3")
            top_donors = cursor.fetchall()
            print(f"   🏆 Top 3 Donors: {top_donors}")

        return final_donation_count > 0

    except Exception as e:
        print(f"❌ Error loading CSV data: {e}")
        import traceback
        traceback.print_exc()
        conn.rollback()
        return False


if __name__ == "__main__":
    print("🏗️  Setting up Donor Analytics Database with Real Data...")
    if create_database():
        print("🎉 Database setup completed successfully!")
        print("\nNext steps:")
        print("1. Run: python rag_system_old.py")
        print("2. Run: streamlit run app.py")
    else:
        print("❌ Database setup failed! Check the errors above.")