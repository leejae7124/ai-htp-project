from db_connect import db
from datetime import datetime

class User(db.Model):
    userid = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), nullable=False)
    image1 = db.Column(db.BLOB)
    image2 = db.Column(db.BLOB)
    crop1_1001 = db.Column(db.BLOB)
    crop1_1002 = db.Column(db.BLOB)
    crop1_1003 = db.Column(db.BLOB)
    crop1_1004 = db.Column(db.BLOB)
    crop2_1001 = db.Column(db.BLOB)
    crop2_1002 = db.Column(db.BLOB)
    crop2_1003 = db.Column(db.BLOB)
    crop2_1004 = db.Column(db.BLOB)
    result1 = db.Column(db.TEXT)
    result2 = db.Column(db.TEXT)

class EntireTree(db.Model):
	id = db.Column(db.Integer, primary_key=True)
	result = db.Column(db.TEXT)
        
class Root(db.Model):
	id = db.Column(db.Integer, primary_key=True)
	result = db.Column(db.TEXT)
	
class Branch(db.Model):
	id = db.Column(db.Integer, primary_key=True)
	result = db.Column(db.TEXT)
	
class Leap(db.Model):
	id = db.Column(db.Integer, primary_key=True)
	result = db.Column(db.TEXT)
	
class Stem(db.Model):
	id = db.Column(db.Integer, primary_key=True)
	result = db.Column(db.TEXT)
	
class Size(db.Model):
	id = db.Column(db.Integer, primary_key=True)
	result = db.Column(db.TEXT)