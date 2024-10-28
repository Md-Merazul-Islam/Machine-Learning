from pymongo import MongoClient
MONGO_URL ="mongodb://localhost:27017/?directConnection=true"


conn = MongoClient(MONGO_URL)