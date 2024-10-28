def noteEntity(item)-> dict:
  return{
    'id':str(item["_id"]),
    'name': item["name"],
    'desc': item["desc"],
    'important': item["important"],
  }
  
  
  
def notesEntity(items)-> dict:
  return [noteEntity(item) for item in items]