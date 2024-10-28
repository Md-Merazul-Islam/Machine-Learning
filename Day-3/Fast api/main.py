from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pymongo import MongoClient

app = FastAPI()

# Correct MongoDB connection string
con = MongoClient("mongodb://localhost:27017/?directConnection=true")
db = con.mydb  # Specify the database here

# Mount the static files and set up templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
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

@app.get("/admin")
def get_admin():
    return {"message": "Welcome to the admin section"}

@app.post("/admin")
def create_admin(username: str, password: str):
    # validate the username and password here
    return {"message": "Admin created successfully"}
