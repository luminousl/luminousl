from sqlalchemy.orm import Session
import datetime
from . import models, schemas

def find_file_by_id(db: Session, file_id: int):
    return db.query(models.File).filter(models.File.id == file_id).first()

def find_file_by_signature(db: Session, signature: str):
    return db.query(models.File).filter(models.File.signature == signature).first()

# def find_files_by_session(db: Session, session: str):
#     return db.query(models.File).filter(models.File.session == session).all()

# def find_files_by_source_and_session(db: Session, source: str, session: str):
#     return db.query(models.File).filter(models.File.session == session).filter(models.File.source == source).all()

def create_file(db: Session, new_file: schemas.FileBase):
    # Check if file with same signature already exists
    if new_file.signature:
        existing_file = db.query(models.File).filter(models.File.signature == new_file.signature).first()
        if existing_file:
            return existing_file

    file = models.File(
        name=new_file.name,
        virtual_folder=new_file.virtual_folder,
        signature=new_file.signature,
        description=new_file.description,
        size_bytes=new_file.size_bytes,
        create_time=datetime.datetime.now(),
        update_time=datetime.datetime.now(),
    )
    db.add(file)
    db.flush([file])
    return file

def find_view_by_id(db: Session, view_id: int):
    return db.query(models.View).filter(models.View.id == view_id).first()

def find_view_by_keyword(db: Session, keyword: str, page: int, pagesize: int):
    cond = db.query(models.View).filter(models.View.name.contains(keyword) | models.View.virtual_folder.contains(keyword))
    return cond.order_by(models.View.update_time.desc()).slice(page * pagesize, (page + 1) * pagesize).all(), cond.count()

def all_views(db: Session):
    return db.query(models.View).all()

def create_view(db: Session, new_view: schemas.ViewBase):
    view = models.View(
        name=new_view.name,
        virtual_folder=new_view.virtual_folder,
        view_type=new_view.view_type,
        meta_data=new_view.meta_data,
        create_time=datetime.datetime.now(),
        update_time=datetime.datetime.now(),
    )
    db.add(view)
    db.flush([view])
    return view

def find_issues_by_view_id(db: Session, view_id: int, status: str):
    return db.query(models.Issue).filter(models.Issue.view_id == view_id).filter(models.Issue.status == status).all()

def find_issue_by_id(db: Session, issue_id: int, status: str):
    return db.query(models.Issue).filter(models.Issue.id == issue_id).filter(models.Issue.status == status).first()

def create_issue(db: Session, new_issue: schemas.IssueBase):
    issue = models.Issue(
        keywords=new_issue.keywords,
        description=new_issue.description,
        view_id=new_issue.view_id,
        status="normal",
        creator=new_issue.creator,
        associate_nodes=new_issue.associate_nodes,
        create_time=datetime.datetime.now(),
        update_time=datetime.datetime.now(),
    )
    db.add(issue)
    db.flush([issue])
    return issue
