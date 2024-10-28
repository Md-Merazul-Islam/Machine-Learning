from fastapi import APIRouter, Request, FastAPI
from models.note import Note
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pymongo import MongoClient

from config.db import conn

from schemas.note import noteEntity, notesEntity
app = FastAPI()
note = APIRouter()
# Mount the static files and set up templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@note.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    docs = db.my.find({})
    newdata = []
    for doc in docs:
        newdata.append({
            'id': doc['_id'],
            "name": doc['name'],
        })
        print(doc)
    return templates.TemplateResponse(
        "index.html", {"request": request, "newdata": newdata}
    )




@note.get("/admin")
def get_admin():
    return {"message": "Welcome to the admin section"}

@note.post("/admin")
def create_admin(username: str, password: str):
    # validate the username and password here
    return {"message": "Admin created successfully"}
