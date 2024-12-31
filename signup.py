import pyodbc
from flask import Flask, jsonify,request
import FYPAPIs.app as app
@app.route('/DoctorSignup', methods=['POST'])
def add_doctor():
    user = request.get_json()
    cursor.execute('SELECT * FROM [User] WHERE UserID=?', (user['email'],))
    existing_user = cursor.fetchone()
    if existing_user:
        return jsonify({"status":"User with the same email already exist"}),409
    cursor.execute('insert into [user] values(?,?,?,?,?,?,?)',(user['email'],user['name'],user['password'],user['gender'],user['age'],user['role'],user['imagepath']))
    cursor.execute('select * from [User] where name=? and UserID=?',user["name"],user["email"])
    result=cursor.fetchone()
    if result:
        cursor.execute('insert into doctor values(?)',user['email'])
        conn.commit()
        return jsonify({"status":"SignUp"}),200
    else:
        conn.rollback()
    return jsonify({"status":"SignUp Failed"}),201
    

@app.route('/PatientSignup', methods=['POST'])
def add_patient():
    user = request.get_json()
    cursor.execute('SELECT * FROM [User] WHERE UserID=?', (user['email'],))
    existing_user = cursor.fetchone()
    if existing_user:
        return jsonify({"status":"User with the same email already exist"}),409
    cursor.execute('insert into [user] values(?,?,?,?,?,?,?)',(user['email'],user['name'],user['password'],user['gender'],user['age'],user['role'],user['imagepath']))
    cursor.execute('select * from [User] where name=? and UserID=?',user["name"],user["email"])
    result=cursor.fetchone()
    if result:
        cursor.execute('insert into patient values(?,?,?)',user["weight"],user["height"],user['email'])
        conn.commit()
        return jsonify({"status":"SignUp"}),200
    else:
        conn.rollback()
        return jsonify({"status":"SignUp Failed"}),201

@app.route('/SupervisorSignup', methods=['POST'])
def add_supervisor():
    user = request.get_json()
    cursor.execute('SELECT * FROM [User] WHERE UserID=?', (user['email'],))
    existing_user = cursor.fetchone()
    if existing_user:
        return jsonify({"status":"User with the same email already exist"}),409
    cursor.execute('insert into [user] values(?,?,?,?,?,?,?)',(user['email'],user['name'],user['password'],user['gender'],user['age'],user['role'],user['imagepath']))
    cursor.execute('select * from [User] where name=? and UserID=?',user["name"],user["email"])
    result=cursor.fetchone()
    if result:
        cursor.execute('insert into Supervisor values(?)',user['email'])
        conn.commit()
        return jsonify({"status":"SignUp"}),200
    else:
        conn.rollback()
    return jsonify({"status":"SignUp Failed"}),201
        
