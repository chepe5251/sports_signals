import pyodbc
from src.utils.config import (
    DB_SERVER,
    DB_NAME,
    DB_TRUSTED,
   
)

def get_connection():
    """
    Devuelve una conexión activa a SQL Server
    """
    if DB_TRUSTED:
        conn_str = (
            "DRIVER={ODBC Driver 17 for SQL Server};"
            f"SERVER={DB_SERVER};"
            f"DATABASE={DB_NAME};"
            "Trusted_Connection=yes;"
        )
    else:
        conn_str = (
            "DRIVER={ODBC Driver 17 for SQL Server};"
            f"SERVER={DB_SERVER};"
            f"DATABASE={DB_NAME};"
            
        )

    return pyodbc.connect(conn_str)
