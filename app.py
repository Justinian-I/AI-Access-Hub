import json
from bson import ObjectId,json_util
from flask import Flask,request,jsonify,render_template,session, redirect, url_for,flash
from datetime import datetime,timedelta
from uuid import uuid4
import re
from mon_connect import collection,database,client
from VarifiMon import validate, validateLogin
import os
from datetime import datetime
from subprocess import run
from tensorflow.keras.models import load_model
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import autokeras as ak
from flask import send_from_directory



app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.permanent_session_lifetime = timedelta(days=1)
# Path to temporarily save uploaded images
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Store the loaded model globally (to avoid reloading for every request)
loaded_model = None
loaded_classes = []

@app.route('/load_model', methods=['POST'])
def load_model_route():
    global loaded_model, loaded_classes

# Load the model
    model_directory = request.form['model_directory']
    model_name = request.form['model_name']
    model_path = os.path.join(model_directory, f"{model_name}.keras")

    if os.path.exists(model_path):
        # Correct usage of `load_model`
        loaded_model = load_model(model_path, custom_objects=ak.CUSTOM_OBJECTS)
        print("Model loaded successfully.")

        
        
        # Retrieve classes from the model name or database (if stored)
        models_collection = database['models_collection']
        model_data = models_collection.find_one({"model_directory": model_directory}, {"_id": 0, "classes": 1})
        loaded_classes = model_data['classes'] if model_data else []

        return render_template('interact_with_model.html', model_name=model_name, classes=loaded_classes,model_directory=model_directory)
    else:
        return jsonify({"error": "Model file not found."}), 404

@app.route('/download_model', methods=['GET'])
def download_model():
    model_directory = request.args.get('model_directory')  # Relative directory from MongoDB
    model_name = request.args.get('model_name')  # Model name without extension

    if not model_directory or not model_name:
        return jsonify({"error": "Missing model directory or model name"}), 400

    keras_file_name = f"{model_name}.keras"  # Name of the file to serve
    absolute_model_directory = os.path.abspath(model_directory)  # Convert to absolute path

    # Debugging: Log the absolute directory and file name
    print(f"Model directory: {absolute_model_directory}")
    print(f"File name: {keras_file_name}")

    # Check if the file exists
    keras_file_path = os.path.join(absolute_model_directory, keras_file_name)
    if os.path.exists(keras_file_path):
        print("File exists. Serving file...")
        return send_from_directory(
            directory=absolute_model_directory,  # Path to the directory
            path=keras_file_name,  # Name of the file within the directory
            as_attachment=True,
            mimetype="application/octet-stream"  # Ensures the file is downloaded
        )
    else:
        print(f"File not found: {keras_file_path}")
        return jsonify({"error": f"Model file '{keras_file_name}' not found"}), 404


@app.route('/download_metrics', methods=['GET'])
def download_metrics():
    model_directory = request.args.get('model_directory')

    if not model_directory:
        return jsonify({"error": "Missing model directory"}), 400

    metrics_file_name = "metrics.json"
    absolute_model_directory = os.path.abspath(model_directory)

    metrics_file_path = os.path.join(absolute_model_directory, metrics_file_name)
    if os.path.exists(metrics_file_path):
        return send_from_directory(
            directory=absolute_model_directory,
            path=metrics_file_name,
            as_attachment=True,
            mimetype="application/json"
        )
    else:
        return jsonify({"error": f"Metrics file '{metrics_file_name}' not found"}), 404

@app.route('/classify_image', methods=['POST'])
def classify_image():
    global loaded_model, loaded_classes

    if loaded_model is None:
        return jsonify({"error": "No model loaded."}), 400

    # Handle uploaded file
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected."}), 400

    # Save and preprocess the image
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Preprocess the image
    img_array = preprocess_image(filepath)

    # Predict the class
    predictions = loaded_model.predict(img_array)

    if len(loaded_classes) == 2:
        # Boolean classification (binary classification)
        threshold = 0.5
        predicted_index = 1 if predictions[0][0] >= threshold else 0
        predicted_label = loaded_classes[predicted_index]
        raw_probability = predictions[0][0]  # Get raw probability for class '1'
    else:
        # Multi-class classification
        predicted_index = np.argmax(predictions, axis=1)[0]
        predicted_label = loaded_classes[predicted_index] if loaded_classes else str(predicted_index)
        raw_probability = predictions[0][predicted_index]  # Get probability for the predicted class

    return jsonify({
        "predicted_label": predicted_label
    })
    
# Helper function to preprocess image
def preprocess_image(img_path, target_size=(64, 64)):
    img = Image.open(img_path).convert('L')  # Convert to grayscale
    img = img.resize(target_size)  # Resize to match the model's input size
    img_array = np.array(img).astype('float32') / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=(0, -1))  # Add batch and channel dimensions
    return img_array
    
    

def make_dataset_directory(user, query):
    base_dir = "datasets"  # Base directory to store datasets
    user_dir = os.path.join(base_dir, user)  # User-specific directory
    dataset_dir = os.path.join(user_dir, query.replace(' ', '_'))  # Dataset-specific directory

    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    return dataset_dir


def create_model_directory(user, model_name):
    base_dir = "models"
    user_dir = os.path.join(base_dir, user)
    
    # Ensure model_name is a valid string
    if model_name is None:
        model_name = "default_model"  # Or any other fallback name
    
    model_dir = os.path.join(user_dir, model_name)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    return model_dir



@app.route('/home')
def home():
    if 'user' in session:
        return render_template('home.html', logged_in=True)
    return render_template('home.html', logged_in=False)

@app.route('/signup', methods=['GET'])
def signup_page():
    return render_template('signup.html')

@app.route('/',methods=['GET'])
def index():
    return render_template('home.html')
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        json_data = request.get_json()
        name = json_data.get('name')
        username = json_data.get('username')
        email = json_data.get('email')
        password = json_data.get('password')

        eventid = datetime.now().strftime('%Y%m-%d%H-%M%S-') + str(uuid4())
        fetch = {"femail": email, "fusername": username}
        info = {
            "_id": eventid,
            "fname": name,
            "fusername": username,
            "femail": email,
            "fpassword": password
        }

        # Validate the input
        fieldValidation = validate(info)
        if fieldValidation:
            # Check if the user already exists
            existing_user = collection.find_one(fetch)
            if not existing_user:
                # Save the user to the database
                collection.insert_one(info)

                # Set the user in the session
                session['user'] = username

                # Redirect to the home page
                return jsonify({"message": f"WELCOME! {name}", "redirect": "/home"})
            else:
                return jsonify({"message": "User already exists!"})
        return jsonify({"message": "Validation failed!"})

    return render_template('signup.html')
    
@app.route('/progress', methods=['GET'])
def progress():
    def generate():
        yield "data: Starting scraping...\n\n"
        # Simulate scraping
        yield "data: Scraping complete.\n\n"
        yield "data: Starting preprocessing...\n\n"
        # Simulate preprocessing
        yield "data: Preprocessing complete.\n\n"
        yield "data: Training model...\n\n"
        # Simulate training
        yield "data: Training complete.\n\n"
        yield "data: Done.\n\n"
    return Response(generate(), content_type='text/event-stream')

@app.route('/profile')
def profile():
    if 'user' not in session:
        return redirect(url_for('login'))

    user = collection.find_one(
        {"fusername": session['user']},
        {"_id": 0, "fusername": 1, "fpassword": 1, "femail": 1})
    if user:
        return render_template('profile.html', user=user, logged_in=True)
    return redirect(url_for('home'))
    
@app.route('/submit_model', methods=['POST'])
def submit_model():
    if 'user' not in session:
        return redirect(url_for('login'))

    query = request.form['query']
    selected_classes = request.form.getlist('classes')
    if query:
        selected_classes.insert(0, query)
    user = session['user']
    date_created = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    model_name = f"{'_'.join(selected_classes)}"
    model_directory = create_model_directory(user, model_name)

    progress = {"status": "Starting"}

    try:
        # Step 1: Check for existing data
        data_ready = True
        if query:
            query_path = os.path.join('C:\\Users\\Mohammad\\Graduation_Project\\pre_data', query)
            if not os.path.exists(query_path) or not os.listdir(query_path):
                data_ready = False

        # Step 2: Scraping (if needed)
        if not data_ready and query:
            progress["status"] = "Scraping in progress"
            scraping_result = run(["python", "Scraping.py", query], capture_output=True, text=True)
            if scraping_result.returncode != 0:
                raise Exception(f"Scraping failed: {scraping_result.stderr}")

        # Step 3: Preprocessing (if needed)
        if not data_ready:
            progress["status"] = "Preprocessing in progress"
            preprocessing_result = run(["python", "preprocess_the_images.py"], capture_output=True, text=True)
            if preprocessing_result.returncode != 0:
                raise Exception(f"Preprocessing failed: {preprocessing_result.stderr}")

        # Step 4: Training
        progress["status"] = "Training in progress"
        result = run(
            [
                "python", "train_model.py",
                "--classes", ",".join(selected_classes),
                "--user", user,
            ],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise Exception(f"Training failed: {result.stderr}")

        # Step 5: Parse metrics from training
        metrics_path = os.path.join(model_directory, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
        else:
            metrics = {"val_loss": None, "val_accuracy": None}
        if (len(selected_classes) == 1):
            selected_classes.insert(0, "other")
            
        # Step 6: Save model details
        model_doc = {
            "user": user,
            "model_name": model_name,
            "classes": selected_classes,
            "date_created": date_created,
            "model_directory": model_directory,
            "metrics": metrics,
        }
        models_collection = database['models_collection']
        models_collection.insert_one(model_doc)

        progress["status"] = "Training complete"
        flash(f"Model '{model_name}' created successfully.")
        return redirect(url_for('home'))
    except Exception as e:
        return jsonify({"error": str(e), "progress": progress}), 500
        

@app.route('/make_model', methods=['GET'])
def make_model():
    if 'user' not in session:
        return redirect(url_for('login'))

    # Fetch dataset names from MongoDB
    dataset_collection = database["datasets"]  # Adjust 'database' to your actual MongoDB database instance
    existing_datasets = dataset_collection.distinct("dataset_name")  # Fetch unique dataset names

    # Render the template and pass the existing datasets
    return render_template('make_model.html', existing_datasets=existing_datasets)

@app.route('/view_model', methods=['GET'])
def view_model():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    user = session['user']
    models_collection = database['models_collection']
    
    # Query models created by the current user
    models = list(models_collection.find({"user": user}, {"_id": 0}))

    return render_template('view_models.html', models=models)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        desc = {"femail": email, "fpassword": password}

        fieldValidation = validateLogin(desc)

        if fieldValidation:
            user = collection.find_one(desc)
            if user:
                session['user'] = user['fusername']
                return redirect(url_for('home'))
            else:
                return render_template('login.html', error="Invalid credentials.")
        else:
            return render_template('login.html', error="Validation failed.")
    return render_template('login.html')
                                                                  
         
#    return jsonify(condition)
@app.route('/read', methods=['GET'])
def read_record():
    if request.method == 'GET':
        response = collection.find()

    return jsonify(list(response))


@app.route('/delete', methods=['POST'])
def delete_record():
   if request.method == 'POST':
      email = request.json['email']

      desc = {"femail":email}
      print(desc)
      del_rec = collection.delete_many(desc)
      response = collection.find()

   return jsonify(list(response))
#create update route post & get method  
@app.route('/update', methods=['GET','POST'])
#create a function
def update_record():
    condition = False
    #checking method
    if request.method == 'POST':
        #get data in json format
        json = request.get_json()
        
        #used for fetching the data in database using name
        myquery = {"fname":request.json['name']}
        #find the data and assign to res variable
        res = collection.find_one(myquery)
        #if there is no data
        if res == None:
            #give a message
            condition = {"message":"no data found"}
        else:
            #if there is data then update
            new_val = {
                "$set":{"fname": json['name'],"fusername":json['username'],"femail":json['email'],"fpassword":json['password']}}
            
            #taking updated dictionary
            newdict = {"fname": json['name'],"fusername":json['username'],"femail":json['email'],"fpassword":json['password']}
            
            #update the record
            new_record = collection.update_one(myquery,new_val)
            #give a message with updated dict
            condition = {"Updated":(newdict)}

        response = collection.find(myquery)
    return jsonify(condition)

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('home'))

if __name__ == "__main__":
   app.run(debug=True)