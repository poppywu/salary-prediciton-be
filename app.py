from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from transformers import pipeline
import joblib

app = Flask(__name__)

# Hugging Face model for skills extraction
extractor = pipeline('ner', model="algiraldohe/lm-ner-linkedin-skills-recognition")

# # Load the Keras model correctly
model = tf.keras.models.load_model('salary_prediction_model.keras')

@app.route('/api/predict-salary', methods=['POST'])
def predict_salary():
    data = request.json
    print(data)
    title = data.get('title')
    industry = data.get('industry')
    experience = data.get('experience')
    location = data.get('location')
    description = data.get('description')

    if not all([title, industry, experience, location, description]):
        return jsonify({'error': 'All fields are required'}), 400

    # Prepare data for salary prediction
    final_input, skills_dict = process_input(title, location, industry, experience, description)
    print(skills_dict)

    # Predict salary log
    predicted_salary_log = model.predict(final_input)[0][0]  # Extract the value
    
    # Transform the predicted log-salary back to the original scale
    predicted_salary = np.expm1(predicted_salary_log)

    print(predicted_salary)
    return jsonify({'predictedSalary': round(float(predicted_salary),2), 'extractedSkills': skills_dict})

def process_input(title, location, industry, experience, description):
    # Load encoders and vectorizers
    title_encoder = joblib.load('title_encoder.pkl')
    industry_encoder = joblib.load('industry_encoder.pkl')
    experience_encoder = joblib.load('experience_encoder.pkl')
    location_encoder = joblib.load('location_encoder.pkl')
    tfidf_encoder = joblib.load('tfidf_encoder.pkl')
    scaler = joblib.load('scaler.pkl')

    # Transform inputs using pre-fitted encoders
    title_encoded = title_encoder.transform([title])[0]
    industry_encoded = industry_encoder.transform([industry])[0]
    experience_encoded = experience_encoder.transform([experience])[0]
    location_encoded = location_encoder.transform([location])[0]

    # Extract skills
    skills_dict = skills_extraction(description)
    technical_skill = skills_dict.get('technical_skill', '')
    
    # Transform skills using TF-IDF
    tfidf_tech_skills = tfidf_encoder.transform([technical_skill]).toarray()

    # Concatenate all features (ensure correct shape)
    combined_features = np.array([[title_encoded, location_encoded, industry_encoded, experience_encoded]])
    final_input = np.concatenate([combined_features, tfidf_tech_skills], axis=1)
    final_input = scaler.transform(final_input)  # Apply scaling
    return final_input, skills_dict

def skills_extraction(description):
    extracted_skills = extractor(description)
    raw_skills = process_entities(extracted_skills)
    skills_dict = separate_skills(raw_skills)
    return skills_dict

def process_entities(entities, score_threshold=0.7):
    skills = []
    current_skill = []
    current_type = None

    for ent in entities:
        word = ent['word']
        score = ent['score']
        entity_type = ent['entity'].split('-')[-1]

        if word.startswith('##'):
            word = word[2:]

        if score < score_threshold:
            continue

        if ent['entity'].startswith('B-'):
            if current_skill and current_type:
                skills.append((current_type, "".join(current_skill)))
            current_skill = [word]
            current_type = entity_type
        elif ent['entity'].startswith('I-') and current_skill:
            current_skill.append(word)

    if current_skill and current_type:
        skills.append((current_type, "".join(current_skill)))

    return skills

def separate_skills(skills):
    skills_dict = {}
    for skill_type, skill in skills:
        if skill_type in {'TECHNICAL', 'TECHNOLOGY'}:
            skill_type = 'technical_skill'
        elif skill_type == 'BUS':
            skill_type = 'business_skill'
        elif skill_type == 'SOFT':
            skill_type = 'soft_skill'

        if skill_type not in skills_dict:
            skills_dict[skill_type] = set()
        skills_dict[skill_type].add(skill)

    for skill_type in skills_dict:
        skills_dict[skill_type] = ",".join(skills_dict[skill_type])

    return skills_dict

if __name__ == '__main__':
    app.run(debug=True)
