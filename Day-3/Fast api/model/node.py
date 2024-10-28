from pydantic import BaseModel

class Note(BaseModel):
  name : str
  desc : str 
  important : bool 
  