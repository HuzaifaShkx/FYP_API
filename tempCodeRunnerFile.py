@app.route('/getSupervisorDoctor/<int:sup_id>')
def getSupDoctor(sup_id):
    try:
        cursor=conn.cursor()
        cursor.execute('select doctor_id from DoctorSupervisorRelationship where supervisor_id=?',(sup_id))
        doctor_id=cursor.fetchone()
        if doctor_id:
            d_id={"doctor_id":doctor_id[0]}
            cursor.close()

            return jsonify(d_id),200
        else:
            cursor.close()
    
            return jsonify({"status":"User with this id does not exist"}),405
    except Exception as e:
        return jsonify({"Exception":str(e)}),500
    
