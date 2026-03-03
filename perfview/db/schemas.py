from pydantic import BaseModel

class FileBase(BaseModel):
    virtual_folder: str
    name: str
    signature: str
    description: str
    size_bytes: int

class File(FileBase):
    class Config:
        from_attributes = True

class ViewBase(BaseModel):
    name: str
    virtual_folder: str
    view_type: str
    meta_data: str

class View(ViewBase):
    class Config:
        from_attributes = True

class IssueBase(BaseModel):
    keywords: str
    description: str
    view_id: int
    creator: str
    associate_nodes: str

class Issue(IssueBase):
    class Config:
        from_attributes = True