import sqlite3
import pandas as pd
import numpy as np
import random
from faker import Faker
from datetime import datetime, timedelta

# Initialize Faker
fake = Faker()

# --- 1. DATABASE SETUP ---

def create_tables(cursor):
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS clients (
        client_id TEXT PRIMARY KEY,
        account_creation_date TEXT,
        avg_spend REAL,
        country TEXT,
        ip_address TEXT,
        device_id TEXT,
        has_biometrics INTEGER
    )
    ''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS merchants (
        merchant_id TEXT PRIMARY KEY,
        name TEXT,
        category TEXT,
        risk_score REAL,
        account_creation_date TEXT,
        country TEXT
    )
    ''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS transactions (
        transaction_id TEXT PRIMARY KEY,
        timestamp TEXT,
        amount REAL,
        currency TEXT,
        location TEXT,
        ip_address TEXT,
        device_id TEXT,
        client_id TEXT,
        merchant_id TEXT,
        is_fraud INTEGER,
        FOREIGN KEY(client_id) REFERENCES clients(client_id),
        FOREIGN KEY(merchant_id) REFERENCES merchants(merchant_id)
    )
    ''')

# --- 2. DATA GENERATORS ---

def generate_clients(n=100):
    return [
        (
            fake.uuid4(),
            fake.date_between(start_date='-5y', end_date='-1d').isoformat(),
            round(random.uniform(20, 1000), 2),
            fake.country(),
            fake.ipv4(),
            fake.uuid4(),
            random.choice([0, 1])
        )
        for _ in range(n)
    ]

def generate_merchants(n=50):
    return [
        (
            fake.uuid4(),
            fake.company(),
            random.choice(['Retail', 'Electronics', 'Food', 'Fashion', 'Travel', 'Gaming']),
            round(random.uniform(0, 1), 2),
            fake.date_between(start_date='-10y', end_date='-1d').isoformat(),
            fake.country()
        )
        for _ in range(n)
    ]

def generate_transactions(clients, merchants, n=10000):
    return [
        (
            fake.uuid4(),
            (datetime.now() - timedelta(minutes=random.randint(0, 1000000))).isoformat(),
            round(random.uniform(1, 5000), 2),
            random.choice(['USD', 'EUR', 'GBP', 'MXN']),
            fake.city(),
            fake.ipv4(),
            fake.uuid4(),
            random.choice(clients)[0],
            random.choice(merchants)[0],
            int(np.random.choice([0, 1], p=[0.95, 0.05]))  # Ensure is_fraud is an integer
        )
        for _ in range(n)
    ]

# --- 3. MAIN EXECUTION ---

def main():
    # Connect to SQLite (use ':memory:' for RAM or 'fraud_poc.db' to persist)
    conn = sqlite3.connect('fraud_poc.db')
    cursor = conn.cursor()

    # Create tables
    create_tables(cursor)

    # Generate and insert synthetic data
    client_data = generate_clients()
    merchant_data = generate_merchants()
    transaction_data = generate_transactions(client_data, merchant_data)

    cursor.executemany('INSERT INTO clients VALUES (?, ?, ?, ?, ?, ?, ?)', client_data)
    cursor.executemany('INSERT INTO merchants VALUES (?, ?, ?, ?, ?, ?)', merchant_data)
    cursor.executemany('INSERT INTO transactions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', transaction_data)
    conn.commit()

    # Optional: show sample
    sample_df = pd.read_sql_query("SELECT * FROM transactions LIMIT 5", conn)
    print(sample_df)

    conn.close()

if __name__ == "__main__":
    main()

