from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, DateTime, BigInteger, Boolean
from pydantic import BaseModel
import os

Base = declarative_base()

PERFVIEW_DEBUG = os.getenv("PERFVIEW_DEBUG", "0") == "1"

if PERFVIEW_DEBUG:
    print("Starting in debug mode")
    PERFVIEW_DATABASE_FILE = os.path.abspath(os.getenv("PERFVIEW_DATABASE_FILE", "./debug.db"))
    SQLALCHEMY_DATABASE_URL = f"sqlite:///{PERFVIEW_DATABASE_FILE}"
else:
    PERFVIEW_DB_USER = os.getenv("PERFVIEW_DB_USER", "root")
    PERFVIEW_DB_PASSWORD = os.getenv("PERFVIEW_DB_PASSWORD", "nvcntse")
    PERFVIEW_DB_HOST = os.getenv("PERFVIEW_DB_HOST", "127.0.0.1")
    PERFVIEW_DB_PORT = os.getenv("PERFVIEW_DB_PORT", "3306")
    PERFVIEW_DB_NAME = os.getenv("PERFVIEW_DB_NAME", "trt_perf_view_dev")
    PERFVIEW_DB_CHARSET = os.getenv("PERFVIEW_DB_CHARSET", "utf8mb3")
    SQLALCHEMY_DATABASE_URL = f"mysql+pymysql://{PERFVIEW_DB_USER}:{PERFVIEW_DB_PASSWORD}@{PERFVIEW_DB_HOST}:{PERFVIEW_DB_PORT}/{PERFVIEW_DB_NAME}?charset={PERFVIEW_DB_CHARSET}"

print(f"Database file: {SQLALCHEMY_DATABASE_URL}")

# engine = create_engine(
#     SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False, "timeout": 500}
# )
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_recycle=300,
    pool_pre_ping=True,
    pool_timeout=30
)
Base.metadata.create_all(bind=engine)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class File(Base):
    __tablename__ = "files"

    id = Column(Integer, primary_key=True, index=True)
    virtual_folder = Column(String, index=True)
    name = Column(String, index=True)
    description = Column(String)
    signature = Column(String)
    size_bytes = Column(BigInteger)
    create_time = Column(DateTime)
    update_time = Column(DateTime)

class View(Base):
    __tablename__ = "views"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    virtual_folder = Column(String, index=True)
    view_type = Column(String, index=True)
    meta_data = Column(String)
    create_time = Column(DateTime)
    update_time = Column(DateTime)

class Issue(Base):
    __tablename__ = "issues"

    id = Column(Integer, primary_key=True, index=True)
    keywords = Column(String, index=True)
    creator  = Column(String, index=True)
    status   = Column(String, index=True)
    view_id   = Column(Integer, index=True)
    associate_nodes = Column(String)
    description = Column(String)
    create_time = Column(DateTime)
    update_time = Column(DateTime)