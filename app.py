from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import fitz  # PyMuPDF
from PIL import Image
import os
import google.generativeai as genai
import nltk
import torch
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pymongo import MongoClient




app = Flask(__name__)
CORS(app)

# MongoDB Configuration
client = MongoClient("mongodb+srv://amaresh:1234@artgallery.vntex.mongodb.net/?retryWrites=true&w=majority&appName=artgallery")

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')
# Initialize NLP tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
negation_words = {"not", "never", "no", "none", "cannot", "n't"}  # Add more if needed

# SBERT Model for Similarity
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
genai.configure(api_key="AIzaSyAawh0tRqyCOsyz7x9GxVbV_tkUzBsZ59s")

# Cross-Encoder for Contextual Understanding
cross_encoder_model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/stsb-roberta-large")
cross_encoder_tokenizer = AutoTokenizer.from_pretrained("cross-encoder/stsb-roberta-large")

# Directory to save uploaded files
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# Storage for teacher answers
teacher_answers = {}  # {page_number: text}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Preprocessing Function
def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # Convert to lowercase & tokenize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]  # Remove stopwords & lemmatize
    return " ".join(tokens)

# Function to Check for Negation
def contains_negation(text):
    tokens = set(word_tokenize(text.lower()))
    return any(word in negation_words for word in tokens)

def bert_similarity(student_answer, original_answer):
    # Preprocess text
    student_answer_clean = preprocess_text(student_answer)
    original_answer_clean = preprocess_text(original_answer)

    # SBERT similarity
    emb1 = sbert_model.encode(student_answer_clean, convert_to_tensor=True)
    emb2 = sbert_model.encode(original_answer_clean, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(emb1, emb2).item() * 100  # Convert to percentage

    # Contextual Understanding with Cross-Encoder
    inputs = cross_encoder_tokenizer(student_answer_clean, original_answer_clean, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = cross_encoder_model(**inputs).logits
    context_score = torch.sigmoid(logits).item() * 100  # Convert to percentage

    # Negation Handling
    student_has_negation = contains_negation(student_answer)
    original_has_negation = contains_negation(original_answer)

    if student_has_negation != original_has_negation:  # If one has negation and the other doesn't
        similarity *= 0.5  # Reduce similarity by 50%
        context_score *= 0.5  # Reduce context score by 50%

    return similarity, context_score

def extract_text_from_image(image):

    model = genai.GenerativeModel("gemini-1.5-flash")

    # Send the image to Gemini Vision API
    prompt = "Extract and return the handwritten text from this image:and does not include any extra text in response"
    response = model.generate_content([prompt, image])

    # Return the extracted text
    return response.text.strip()

def extract_text_from_pdf(pdf_path):
    """
    Extracts handwritten text from a multi-page PDF and returns it as a list.
    Each page's text is stored at its corresponding index.
    """
    doc = fitz.open(pdf_path)
    extracted_text_list = []

    for i, page in enumerate(doc):
        print(f"Processing Page {i + 1}...")

        # Convert PDF page to an image
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

        # Extract text from the image
        text = extract_text_from_image(img)
        extracted_text_list.append(text.strip())  # Strip to remove unnecessary spaces

    return extracted_text_list

@app.route('/')
def index():
    """
    Render the homepage with options to upload Teacher and Student PDFs.
    """
    return render_template("index.html")  # Create an HTML form for uploading files

from flask import request, jsonify
import os

@app.route('/mongodb/databases', methods=['GET'])
def get_databases():
    try:
        databases = client.list_database_names()
        return jsonify({"databases": databases}), 200
    except Exception as e:
        print(f"Error fetching databases: {e}")
        return jsonify({"error": "Failed to fetch databases"}), 500


@app.route('/mongodb/collection-data', methods=['POST'])
def get_collection_data():
    try:
        # Extract database name from the request body
        data = request.json
        database_name = data.get("database")

        # Validate the input
        if not database_name:
            return jsonify({"error": "Database name is required"}), 400

        # Access the database and the 'reports' collection
        db = client[database_name]
        collection_name = "reports"
        collection = db[collection_name]

        # Fetch all documents from the collection
        collection_data = list(collection.find({}, {"_id": 0}))  # Exclude '_id' if not required in output

        return jsonify({"collection_name": collection_name, "data": collection_data}), 200
    except Exception as e:
        # Print error details and send a failure response
        print(f"Error fetching collection data: {e}")
        return jsonify({"error": "Failed to fetch collection data"}), 500



@app.route('/upload/teacher', methods=['POST'])
def upload_teacher_pdf():
    """
    Endpoint to upload teacher's PDF and store answers page-wise.
    """
    # Check if the PDF file is in the request
    if 'pdf' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    # Check if the exam name is in the request
    if 'examName' not in request.form:
        return jsonify({"error": "No exam name provided"}), 400

    global exam_name
    exam_name = request.form['examName']  # Extract exam name
    pdf_file = request.files['pdf']
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_file.filename)

    # Save the PDF file 
    pdf_file.save(pdf_path)

    # Extract text from the uploaded PDF (Assuming extract_text_from_pdf is defined)
    global teacher_answers
    teacher_answers = extract_text_from_pdf(pdf_path)

    # You can now handle the exam name however you like (store it, use it for further processing, etc.)
    print(f"Exam Name: {exam_name}")  # Just printing for now

    # Return success response
    return jsonify({
        "message": "Teacher answers uploaded successfully",
        "examName": exam_name,
        "pages": len(teacher_answers)
    })


@app.route('/upload/student', methods=['POST'])
def upload_student_pdf():
    """
    Endpoint to upload a student's PDF, extract answers, and compare with teacher's answers.
    """
    if 'pdf' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    pdf_file = request.files['pdf']
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_file.filename)
    pdf_file.save(pdf_path)

    # Extract text from the uploaded student PDF
    extracted_answers = extract_text_from_pdf(pdf_path)

    # Compare with teacher's answers
    comparisons = {}
    for page, (student_text, teacher_text) in enumerate(zip(extracted_answers, teacher_answers), start=1):
        similarity_score, contextual_score = bert_similarity(student_text, teacher_text)
        sbert_normalized = similarity_score / 100
        cross_encoder_normalized = contextual_score / 100

        # 4. Weightages
        W1 = 0.4  # Weight for SBERT Similarity
        W2 = 0.6  # Weight for Cross-Encoder

        # 5. Final Score (out of 10)
        total_score = 10 * ((W1 * sbert_normalized) + (W2 * cross_encoder_normalized))
        comparisons[page] = {
            "student_text": student_text,
            "teacher_text": teacher_text,
            "similarity_score": similarity_score,
            "contextual_score": contextual_score,
            "total_score": round(total_score,0)
        }


    # Return the comparisons for display
    return render_template("result.html", comparisons=comparisons)

@app.route('/upload/student_api', methods=['POST'])
def upload_student_pdf_api():
    """
    Endpoint to upload a student's PDF, extract answers, and compare with teacher's answers.
    """
    student_name = request.form.get('studentName')
    roll_number = request.form.get('rollNumber')

    if 'pdf' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    pdf_file = request.files['pdf']
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_file.filename)
    pdf_file.save(pdf_path)

    # Extract text from the uploaded student PDF
    extracted_answers = extract_text_from_pdf(pdf_path)

    # Compare with teacher's answers
    comparisons = {}
    for page, (student_text, teacher_text) in enumerate(zip(extracted_answers, teacher_answers), start=1):
        similarity_score, contextual_score = bert_similarity(student_text, teacher_text)
        total_score = similarity_score*0.5 + contextual_score*0.5
        total_score = total_score/10
        comparisons[page] = {
            "student_text": student_text,
            "teacher_text": teacher_text,
            "similarity_score": similarity_score,
            "contextual_score": contextual_score,
            "total_score": round(total_score,0)
        }


    # Return the comparisons for display
    return jsonify({
        "student_name": student_name,
        "roll_number": roll_number,
        "comparisons": comparisons
    })

@app.route("/save-report", methods=["POST"])
def save_report():
    try:
        data = request.json
        # Create/access a database named after the exam name
        db = client[exam_name]
        collection = db["reports"]  # Use a "reports" collection for the data

        # Insert the report data into MongoDB
        collection.insert_one(data)
        return jsonify({"message": f"Report saved successfully in {exam_name} database!"}), 200
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Failed to save the report"}), 500


@app.route("/reset/teacher", methods=["GET"])
def reset_teacher():
    global teacher_answers
    teacher_answers = None
    return jsonify({"message": "Teacher answers reset successfully"})


if __name__ == '__main__':
    app.run(debug=True)
