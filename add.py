import os
import sqlite3
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration for file uploads
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Database setup
DATABASE = 'database.db'

def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/adopt_add')
def adopt_add():
    return render_template('adopt_add.html')

@app.route('/submit', methods=['POST'])
def submit():
    pet_name = request.form['petName']
    owner_name = request.form['ownerName']
    file = request.files['petImage']
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        image_url = f'/uploads/{filename}'
        
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('INSERT INTO store (petName, ownerName, imageUrl) VALUES (?, ?, ?)', (pet_name, owner_name, image_url))
        conn.commit()
        conn.close()
    
    return redirect(url_for('display'))


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
