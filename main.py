import os
import base64
import subprocess
import cv2
import sqlite3
from flask import Flask, render_template, request, redirect, url_for, session, flash, send_from_directory
from werkzeug.utils import secure_filename
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, ValidationError
import bcrypt
from datetime import datetime
import ml  # Assuming ml is a module with dog_responses and predict_image

app = Flask(__name__, template_folder='templates')

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

# Secret key for CSRF protection
app.config['SECRET_KEY'] = 'abcdefghijklmno'  # Replace with your secret key

# Forms
class SignInForm(FlaskForm):
    username = StringField("Username", validators=[DataRequired()])
    password = PasswordField("Password", validators=[DataRequired()])
    submit = SubmitField("Sign In")

class LoginForm(FlaskForm):
    username = StringField("Username", validators=[DataRequired()])
    password = PasswordField("Password", validators=[DataRequired()])
    submit = SubmitField("Login")

    def validate_username(self, field):
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM login WHERE username=?", (field.data,))
        user = cursor.fetchone()
        conn.close()
        if not user:
            raise ValidationError('Invalid username')
        if bcrypt.checkpw(self.password.data.encode('utf-8'), user['password'].encode('utf-8')) is False:
            raise ValidationError('Invalid password')

# Routes
@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/home')
def home():
    if 'username' in session:
        background_image_url = url_for('static', filename='image1.jpeg')
        return render_template('index.html', background_image_url=background_image_url)
    else:
        return redirect(url_for('login'))

@app.route('/login', methods=['POST', 'GET'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        session['username'] = form.username.data
        flash('You were successfully logged in')
        return redirect(url_for('home'))
    return render_template('login.html', form=form)

@app.route('/signin', methods=['GET', 'POST'])
def signin():
    if 'username' in session:
        return redirect(url_for('home'))
    else:
        form = SignInForm()
        if form.validate_on_submit():
            username = form.username.data
            password = form.password.data
            hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            conn = get_db()
            cursor = conn.cursor()
            cursor.execute("SELECT username FROM login WHERE username=?", (username,))
            if cursor.fetchone():
                raise ValidationError("Username Already Present")
            cursor.execute("INSERT INTO login (username, password) VALUES (?, ?)", (username, hashed_password))
            conn.commit()
            conn.close()
            return redirect(url_for('login'))
        return render_template('register.html', form=form)

@app.route('/identify')
def identify():
    if 'username' in session:
        return render_template("form.html")
    else:
        return redirect(url_for('login'))

@app.route('/grooming')
def grooming():
    if 'username' in session:
        return render_template("grooming/groom.html")
    else:
        return redirect(url_for('login'))
@app.route('/petcare')
def petcare():
    if 'username' in session:
        return render_template("maps.html")
    else:
        return redirect(url_for('login'))
@app.route('/adoption')
def adoption():
    if 'username' in session:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM store')
        pets = cursor.fetchall()
        conn.close()
        return render_template("adopt.html", pets=pets)
    else:
        return redirect(url_for('login'))

@app.route('/submit_adoption_form', methods=['POST'])
def submit_adoption_form():
    name = request.form['name']
    contact = request.form['contact']
    location = request.form.get('location', '')
    responsibility = request.form['responsibility']
    
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('INSERT INTO adopt (name, contact, location, responsibility) VALUES (?, ?, ?, ?)', 
                   (name, contact, location, responsibility))
    conn.commit()
    conn.close()
    
    return redirect(url_for('home'))

@app.route('/adopt_add')
def adopt_add():
    return render_template('adopt_add.html')

@app.route('/submit', methods=['POST'])
def submit():
    pet_name = request.form['petName']
    owner_name = request.form['ownerName']
    pet_age = request.form['petAge']
    pet_breed = request.form['petBreed']
    pet_gender = request.form['petGender']
    pet_vaccinated = request.form['petVaccinated']
    pet_state = request.form['petState']
    pet_city = request.form['petCity']
    contact = request.form['contact']  # Correct key
    file = request.files['petImage']
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        image_url = f'/uploads/{filename}'  # Define image_url here
        print(pet_name, owner_name, pet_age, pet_breed, pet_gender, pet_vaccinated, pet_state, pet_city, contact, image_url)
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO store (petName, ownerName, age, breed, gender, vaccinated, state, city, contact, imageUrl)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (pet_name, owner_name, pet_age, pet_breed, pet_gender, pet_vaccinated, pet_state, pet_city, contact, image_url))
        conn.commit()
        conn.close()
    
    return redirect(url_for('adoption'))


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/vans')
def vans():
    if 'username' in session:
        return render_template("grooming/try1.html")
    else:
        return redirect(url_for('login'))

@app.route('/success', methods=['POST'])
def success():
    if 'username' in session:
        if request.method == 'POST':
            f = request.files['file']
            if f.filename:
                f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
                full_filename = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
                img_path = full_filename

                txt = ml.predict_image(img_path)
                final_text = 'Results from Input Image'
                return render_template("success.html", name=final_text, img=full_filename, out_1=txt)
            else:
                return render_template("form.html")
    else:
        return redirect(url_for('login'))

@app.route('/capture', methods=['GET', 'POST'])
def capture():
    if 'username' in session:
        if request.method == 'POST':
            image_data = request.form['image_data']
            image_bytes = base64.b64decode(image_data.split(',')[1])
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'captured_image.jpg')
            with open(img_path, 'wb') as f:
                f.write(image_bytes)

            txt = ml.predict_image(img_path)
            final_text = 'Results from Input Image'

            return render_template("success.html", name=final_text, img=img_path, out_1=txt)
    else:
        return redirect(url_for('login'))

@app.route('/camera', methods=['GET', 'POST'])
def camera():
    if 'username' in session:
        return render_template("camera.html")
    else:
        return redirect(url_for('login'))

@app.route('/chatbot')
def chatbot():
    if 'username' in session:
        # Start the Chainlit server if not already running
        subprocess.Popen(["python3", "model.py"])
        return redirect("http://127.0.0.1:8000")  # Assuming Chainlit runs on port 8000
    else:
        return redirect(url_for('login'))

@app.route('/petshop')
def petshop():
    if 'username' in session:
        return render_template("petshop.html")
    else:
        return redirect(url_for('login'))

@app.route('/petfood')
def petfood():
    if 'username' in session:
        return render_template('petfood.html')
    else:
        return redirect(url_for('login'))

@app.route('/cart')
def cart():
    if 'username' in session:
        return render_template('cart.html')
    else:
        return redirect(url_for('login'))

@app.route('/toys')
def toys():
    if 'username' in session:
        return render_template('toys.html')
    else:
        return redirect(url_for('login'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl

DB_FAISS_PATH = 'vectorstore/db_faiss'

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

# Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                           chain_type='stuff',
                                           retriever=db.as_retriever(search_kwargs={'k': 2}),
                                           return_source_documents=True,
                                           chain_type_kwargs={'prompt': prompt})
    return qa_chain

# Loading the model
def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model="llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )
    return llm

# QA Model Function
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    # Enable dangerous deserialization
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

# Output function
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response

# Chainlit code
@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to Petcare ChatBot. What is your query?"
    await msg.update()

    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["result"]
    sources = res["source_documents"]

    if sources:
        answer += f"\nSources:" + str(sources)
    else:
        answer += "\nNo sources found"

    await cl.Message(content=answer).send()

if __name__ == '__main__':
    with get_db() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS store (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                petName TEXT NOT NULL,
                ownerName TEXT NOT NULL,
                age TEXT NOT NULL,
                breed TEXT NOT NULL,
                gender TEXT NOT NULL,
                vaccinated TEXT NOT NULL,
                state TEXT NOT NULL,
                city TEXT NOT NULL,
                contact TEXT NOT NULL,
                imageUrl TEXT NOT NULL
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS login (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                password TEXT NOT NULL
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS adopt (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                contact TEXT NOT NULL,
                location TEXT,
                responsibility TEXT NOT NULL
            )
        ''')
    app.run(host="127.0.0.1", port=8080, debug=True)
