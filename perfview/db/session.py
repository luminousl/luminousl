from .models import SessionLocal
from fastapi import Depends
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy import text
import time

def get_database_instance():
    db = SessionLocal()

    # max_retries = 3
    # retry_count = 0
    # retry_delay = 0.5

    # while retry_count < max_retries:
    #     db = SessionLocal()
    #     try:
    #         # Test the connection by executing a simple query
    #         db.execute(text("SELECT 1"))
    #         break
    #     except (OperationalError, SQLAlchemyError):
    #         db.close()
    #         retry_count += 1
    #         if retry_count >= max_retries:
    #             raise
    #         time.sleep(retry_delay)
    #         retry_delay *= 2

    try:
        yield db
    finally:
        db.close()

default_db_depends = Depends(get_database_instance)