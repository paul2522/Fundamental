import json

person = {
      "first name" : "Yuna",
      "last name" : "Jung",
      "age" : 33,
      "nationality" : "South Korea",
      "education" : [{"degree":"B.S degree", "university":"Daehan university", "major": "mechanical engineering", "graduated year":2010}]
       } 

with open("person.json", "w") as f:
    json.dump(person , f)

print("슝~")

with open("person.json", "r", encoding="utf-8") as f:
    contents = json.load(f)
    print(contents["first name"])
    print(contents["education"])