from flask import Flask, render_template, request, send_file, jsonify
import cv2
import numpy as np
import joblib
from PIL import Image
import os
import random
from fpdf import FPDF
import datetime
import base64
import io
from io import BytesIO
from deep_translator import GoogleTranslator # Corrected import for the chatbot


# Initialize Flask app
app = Flask(__name__)

# Load the model
# Ensure 'model.pkl' is in the same directory as your app.py
try:
    model = joblib.load('model.pkl')
except FileNotFoundError:
    print("Error: model.pkl not found. Please ensure the model file is in the same directory as app.py")
    # You might want to handle this more gracefully, e.g., exit or serve an error page
    exit()

# Healthy quotes
quotes = [
    "A healthy outside starts from the inside.",
    "Your body hears everything your mind says — stay positive.",
    "Eat well, move more, feel strong.",
    "Healing begins with awareness.",
    "Health is the crown on the well person’s head.",
    "Health is wealth. Protect your mind.",
    "Early detection saves lives.",
    "A healthy outside starts from the inside.",
    "Wellness begins with awareness."
]

# Doctor suggestions
doctors = [
    {"name": "Dr. Aarthi Ramesh", "hospital": "Apollo Hospitals, Chennai"},
    {"name": "Dr. Vikram Shetty", "hospital": "Manipal Hospitals, Bangalore"},
    {"name": "Dr. Rajesh Iyer", "hospital": "AIIMS, Delhi"},
    {"name": "Anita Raj", "hospital": "Apollo Hospitals", "location": "Chennai"},
    {"name": "Ravi Kumar", "hospital": "AIIMS", "location": "New Delhi"},
    {"name": "Sahana N", "hospital": "CMC", "location": "Vellore"},
    {"name": "Dev Sharma", "hospital": "Fortis", "location": "Mumbai"}
]

# Configure max content length for uploads
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024  # Limit to 2MB

# Route for index.html
@app.route('/')
def index():
    quote = random.choice(quotes)
    return render_template('index.html', quote=quote)

# Predict Route
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No image uploaded", 400

    file = request.files['image']

    # Read the image file into bytes for base64 encoding later
    scan_bytes = file.read()
    # Reset file pointer to the beginning for PIL to read it
    file.seek(0)

    # Process image for prediction
    # Ensure this part matches how your model was trained (e.g., color channels)
    try:
        img = Image.open(file).convert('L').resize((100, 100)) # Convert to grayscale ('L')
        img_array = np.array(img).flatten().reshape(1, -1) # Flatten for your model
        pred = model.predict(img_array)[0]
    except Exception as e:
        print(f"Error processing image for prediction: {e}")
        return "Error processing image for prediction.", 500


    # Encode the original scan image to base64 for display in HTML and PDF
    # We assume the input image is typically JPEG, but you might want to infer type
    scan_base64 = "data:image/jpeg;base64," + base64.b64encode(scan_bytes).decode('utf-8')

    if pred == 1:
        result = "Tumor Detected"
        suggestions = {
            "do": ["Eat leafy greens", "Exercise regularly", "Drink plenty of water"],
            "dont": ["Avoid processed meats", "Limit sugar", "Do not smoke or drink"],
            "eat": ["Fruits", "Omega-3 rich foods", "Turmeric"],
            "avoid": ["Red meat", "Sugary drinks", "Fried foods"]
        }
        doctor = random.choice(doctors)
    else:
        result = "No Tumor Detected"
        suggestions = {
            "do": ["Maintain healthy lifestyle"],
            "dont": ["Avoid skipping meals"],
            "eat": ["Balanced diet"],
            "avoid": ["Overeating"]
        }
        doctor = None

    return render_template(
        'result.html',
        result=result,
        suggestions=suggestions,
        doctor=doctor, # Pass the doctor dictionary as is
        quote=random.choice(quotes),
        scan_image=scan_base64, # Pass the base64 image data to the template
        current_date=datetime.datetime.now().strftime("%Y-%m-%d"),
        current_time=datetime.datetime.now().strftime("%H:%M:%S")
    )


@app.route('/ask_chatbot', methods=['POST'])
def ask_chatbot():
    data = request.get_json()
    question = data.get('question', '')

    # Simple rule-based answers (you can use NLP/ML later)
    responses = {
        "what is brain tumor": "A brain tumor is an abnormal growth of cells in the brain.",
        "symptoms": "Symptoms include headaches, nausea, vision problems, and seizures.",
        "treatment": "Treatments include surgery, radiation therapy, and chemotherapy.",
        "do's": "Follow your doctor's instructions, eat healthy, stay hydrated.",
        "don'ts": "Avoid stress, unhealthy food, and self-medication.",
        "what to eat": "Fresh fruits, vegetables, lean protein, whole grains.",
        "what to avoid": "Processed foods, sugar, alcohol, tobacco.",
        "is brain tumor cancer": "Some brain tumors are cancerous (malignant), others are noncancerous (benign).",
        "can brain tumor cause memory loss": "Yes, depending on its location, a brain tumor can affect memory and thinking.",
        "how is brain tumor diagnosed": "Brain tumors are diagnosed through MRI, CT scans, and biopsy.",
        "is brain tumor curable": "Some brain tumors can be cured with early detection and proper treatment.",
        "how long can you live with a brain tumor": "Life expectancy depends on the tumor type, location, and treatment.",
        "can stress cause brain tumor": "No, stress does not directly cause brain tumors.",
        "types of brain tumor": "There are over 120 types including gliomas, meningiomas, and pituitary tumors.",
        "is brain tumor painful": "It can cause headaches and pressure in the head.",
        "can brain tumor be hereditary": "Some types may be linked to genetic conditions.",
        "how does chemotherapy help": "It kills or stops the growth of cancerous cells.",
        "can brain tumor affect vision": "Yes, tumors near the optic nerve can cause vision problems.",
        "best hospital for brain tumor treatment": "Top hospitals include CMC, AIIMS, and Apollo in India.",
        "can kids get brain tumors": "Yes, though it's rarer than in adults.",
        "early signs of brain tumor": "Frequent headaches, nausea, blurred vision, and confusion.",
        "can mobile radiation cause brain tumor": "There’s no strong scientific evidence to support this.",
        "how to prevent brain tumor": "Avoiding radiation exposure and leading a healthy life may help.",
        "brain tumor surgery risks": "Includes infection, bleeding, and neurological issues.",
        "how long does brain tumor surgery take": "Usually 3 to 6 hours depending on the case.",
        "recovery time after brain surgery": "It may take weeks to months.",
        "can brain tumor reoccur": "Yes, some tumors can come back."
    }

    lower_q = question.lower()
    answer = "Sorry, I don't know that. Please ask something else."

    for key in responses:
        if key in lower_q:
            answer = responses[key]
            break

    # Detect if input is Tamil and translate answer to Tamil if so
    # A simple check for Tamil Unicode range (adjust as needed)
    is_tamil = any(0x0B80 <= ord(char) <= 0x0BFF for char in question)
    if is_tamil:
        try:
            # Added source='auto' for flexibility
            answer = GoogleTranslator(source='auto', target='ta').translate(answer)
        except Exception as e:
            print(f"Translation error: {e}")
            # Fallback to English if translation fails
            pass

    return jsonify({'answer': answer})

# Generate downloadable PDF report
@app.route('/download_report', methods=['POST'])
def download_report():
    result = request.form.get('result')
    # Correctly retrieve separate doctor name and hospital
    doctor_name = request.form.get('doctor_name')
    doctor_hospital = request.form.get('doctor_hospital')
    scan_image_b64 = request.form.get('scan_image') # This is the full 'data:image/jpeg;base64,...' string
    date = request.form.get('date')
    time = request.form.get('time')

    # Retrieve suggestions as newline-separated strings, then split back to lines for PDF
    dos = request.form.get('dos', '').replace("\\n", "\n")
    donts = request.form.get('donts', '').replace("\\n", "\n")
    eat = request.form.get('eat', '').replace("\\n", "\n")
    avoid = request.form.get('avoid', '').replace("\\n", "\n")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, txt="Brain Tumor Detection Report", ln=True, align='C')
    pdf.ln(5) # Add a small line break

    pdf.set_font("Arial", size=11)
    pdf.cell(200, 8, txt=f"Date of Scan: {date}", ln=True)
    pdf.cell(200, 8, txt=f"Time of Scan: {time}", ln=True)
    pdf.cell(200, 8, txt=f"Detection Result: {result}", ln=True)
    pdf.ln(5) # Add a small line break


    # --- Embed the uploaded scan image ---
    if scan_image_b64 and scan_image_b64.startswith('data:image'):
        # Extract base64 part and decode
        img_data_b64 = scan_image_b64.split(',')[1]
        temp_image_path = "temp_scan_image.png" # Define temp path

        try:
            img_bytes = base64.b64decode(img_data_b64)
            img = Image.open(BytesIO(img_bytes))

            # FPDF sometimes has issues with specific image types or modes.
            # Convert to RGB to ensure compatibility.
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Ensure the image is saved in a format FPDF reliably reads
            img.save(temp_image_path, format="PNG") # Always save as PNG for consistency

            # Add image to PDF
            # Adjust x, y, w parameters as needed for positioning and size
            current_y = pdf.get_y()
            pdf.image(temp_image_path, x=55, y=current_y + 5, w=100) # Centered for A4 page width 210
            pdf.ln(110) # Move cursor down after image, adjust this value based on image height

        except Exception as e:
            print(f"Error embedding image in PDF: {e}")
            pdf.cell(200, 10, txt="[Error loading scan image for PDF]", ln=True)
        finally:
            # Clean up the temporary file, regardless of success or failure
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
    else:
        pdf.cell(200, 10, txt="[Scan image not available for PDF]", ln=True)

    # --- Add Doctor Information ---
    # Only show doctor info if tumor detected AND name/hospital are present
    if result == "Tumor Detected" and doctor_name and doctor_hospital:
        pdf.set_font("Arial", style='B', size=12)
        pdf.cell(200, 10, txt="Recommended Doctor:", ln=True)
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 8, txt=f"Name: {doctor_name}", ln=True)
        pdf.cell(200, 8, txt=f"Hospital: {doctor_hospital}", ln=True)
        pdf.ln(5)
    elif result == "No Tumor Detected":
        pdf.set_font("Arial", style='I', size=11)
        pdf.cell(200, 10, txt="No specific doctor recommendation as no tumor was detected.", ln=True)
        pdf.ln(5)


    # --- Add Suggestions ---
    pdf.set_font("Arial", style='B', size=12)
    pdf.cell(200, 10, txt="Do's:", ln=True)
    pdf.set_font("Arial", size=12)
    # Use multi_cell for multi-line text (which the `\n` will provide)
    pdf.multi_cell(0, 7, dos) # Reduced line height for better spacing

    pdf.set_font("Arial", style='B', size=12)
    pdf.cell(200, 10, txt="Don'ts:", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 7, donts)

    pdf.set_font("Arial", style='B', size=12)
    pdf.cell(200, 10, txt="What to Eat:", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 7, eat)

    pdf.set_font("Arial", style='B', size=12)
    pdf.cell(200, 10, txt="What to Avoid:", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 7, avoid)

    # Convert to BytesIO for Flask
    pdf_bytes = pdf.output(dest='S').encode('latin1')
    pdf_stream = io.BytesIO(pdf_bytes)
    return send_file(pdf_stream, as_attachment=True, download_name="Brain_Tumor_Report.pdf", mimetype='application/pdf')


# Template filter for line breaks (though not explicitly used for form submission now)
@app.template_filter('nl2br')
def nl2br(value):
    return value.replace("\n", "<br>")

# Flask server start
if __name__ == '__main__':
    print("Starting Flask server...")
    # debug=True allows for automatic reloading on code changes and provides more detailed error messages
    app.run(debug=True)