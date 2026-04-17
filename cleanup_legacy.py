import mysql.connector
import os
from dotenv import load_dotenv

# Load from CandidateBackend or AdminBackend if available, 
# but for now we try root with no password if it's local dev
def cleanup_legacy():
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password=""
        )
        cursor = conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS candidates_db.assessments_summary;")
        print("Successfully dropped legacy table candidates_db.assessments_summary")
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Cleanup failed: {e}")

if __name__ == "__main__":
    cleanup_legacy()
