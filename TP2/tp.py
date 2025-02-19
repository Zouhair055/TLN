import spacy
import requests
import xml.etree.ElementTree as ET
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.metrics.distance import edit_distance
import numpy as np

# Charger le modèle NLP pour le français (ou anglais selon les questions utilisées)
nlp = spacy.load("en_core_web_sm")  # Change en 'fr_core_news_sm' si questions en français

print("### PARTIE 1: ANALYSE DES QUESTIONS ###")
# Fonction pour analyser une question
def preprocess_question(question):
    print(f"\nTraitement de la question: {question}")
    tokens = word_tokenize(question)
    tokens = [t.lower() for t in tokens if t.isalnum() and t not in stopwords.words('english')]
    doc = nlp(question)
    named_entities = [(ent.text, ent.label_) for ent in doc.ents]
    print("Tokens:", tokens)
    print("Entités Nommées:", named_entities)
    return tokens, named_entities

# Fonction pour trouver l'entité DBpedia associée à une question
def find_dbpedia_entity(query):
    print(f"\nRecherche de l'entité DBpedia pour: {query}")
    url = f"https://lookup.dbpedia.org/api/search?query={query}&format=JSON"
    headers = {"Accept": "application/json"}
    
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        results = response.json().get("docs", [])
        entity_uri = results[0]["resource"] if results else None
        print(f"Entité DBpedia trouvée: {entity_uri}")
        return entity_uri
    
    print("Aucune entité trouvée.")
    return None

# Charger et traiter le fichier questions.xml
def parse_questions_xml(file_path):
    print("\nLecture du fichier questions.xml...")
    tree = ET.parse(file_path)
    root = tree.getroot()
    questions = []

    for question in root.findall('question'):
        question_text = question.find("string[@lang='en']")
        answers_element = question.find('answers')

        if question_text is not None:
            question_text = question_text.text
        else:
            print("⚠️ Question sans texte trouvé, passage...")
            continue  # Ignore la question si aucun texte n'est trouvé

        if answers_element is not None:
            expected_answer = [ans.find('uri').text for ans in answers_element.findall('answer') if ans.find('uri') is not None]
        else:
            print(f"⚠️ Pas de réponse trouvée pour la question: {question_text}")
            expected_answer = []

        questions.append((question_text, expected_answer))

    print(f"Nombre de questions chargées: {len(questions)}")
    return questions

# Charger les questions
questions_file = "questions.xml"
questions_data = parse_questions_xml(questions_file)

# Traiter chaque question
def process_questions(questions_data):
    for i, (question_text, expected_answer) in enumerate(questions_data[:5]):  # Limite à 5 questions pour affichage
        print(f"\n### Question {i+1} ###")
        tokens, named_entities = preprocess_question(question_text)
        print(f"Réponse attendue: {expected_answer}")
        
        # Essayer de trouver une entité DBpedia
        for entity, label in named_entities:
            uri = find_dbpedia_entity(entity)
            print(f"Entité DBpedia trouvée pour {entity}: {uri}")

process_questions(questions_data)

print("### PARTIE 2: IDENTIFICATION DES ENTITÉS ET RELATIONS ###")
# Fonction pour trouver l'entité DBpedia associée à une question
def find_dbpedia_entity(query):
    print(f"\nRecherche de l'entité DBpedia pour: {query}")
    url = f"https://lookup.dbpedia.org/api/search?query={query}&format=JSON"
    headers = {"Accept": "application/json"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        results = response.json()["docs"]
        entity_uri = results[0]["resource"] if results else None
        print(f"Entité DBpedia trouvée: {entity_uri}")
        return entity_uri
    print("Aucune entité trouvée.")
    return None

print("### PARTIE 3: CRÉATION ET ÉVALUATION DE LA REQUÊTE SPARQL ###")
# Création de la requête SPARQL
def create_sparql_query(subject, relation):
    print(f"\nCréation de la requête SPARQL pour le sujet: {subject} et la relation: {relation}")
    query = f"""
    PREFIX dbo: <http://dbpedia.org/ontology/>
    PREFIX res: <http://dbpedia.org/resource/>
    SELECT DISTINCT ?uri WHERE {{
        res:{subject} dbo:{relation} ?uri .
    }}
    """
    print("Requête SPARQL générée:")
    print(query)
    return query

# Exemple de requête
sparql_query = create_sparql_query("Brooklyn_Bridge", "crosses")

# Fonction d'évaluation

def evaluate_system(predictions, gold_standard):
    print("\nÉvaluation du système")
    tp = sum(1 for p in predictions if p in gold_standard)
    fp = sum(1 for p in predictions if p not in gold_standard)
    fn = sum(1 for g in gold_standard if g not in predictions)
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f_measure = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0
    print(f"Précision: {precision}, Rappel: {recall}, F-mesure: {f_measure}")
    return precision, recall, f_measure

# Exemple d'évaluation
predicted_answers = ["http://dbpedia.org/resource/East_River"]
gold_answers = ["http://dbpedia.org/resource/East_River"]
evaluate_system(predicted_answers, gold_answers)
