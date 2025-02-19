import xml.etree.ElementTree as ET
from nltk.corpus import sentiwordnet as swn
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import os
from nltk.chunk import ne_chunk

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('sentiwordnet')
nltk.download('wordnet')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('maxent_ne_chunker_tab')

# Chargement des ressources NLTK
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('sentiwordnet')
nltk.download('wordnet')

# Fonction pour obtenir la polarité d'un mot
def get_polarity(word, pos=None):
    synsets = list(swn.senti_synsets(word, pos))
    if synsets:
        synset = synsets[0]
        return synset.pos_score() - synset.neg_score()
    return 0

# Prétraitement des phrases avec NER
def preprocess_sentence(text):
    tokens = word_tokenize(text.lower())
    pos_tags = pos_tag(tokens)
    named_entities = ne_chunk(pos_tags)
    return tokens, pos_tags, named_entities

# Détection de négations
def detect_negation(tokens):
    negations = {'not', 'no', 'never', 'none'}
    for i, token in enumerate(tokens):
        if token in negations:
            return True, i
    return False, -1

# Vérification de la présence d'un aspect dans une phrase
def find_aspect_in_sentence(aspect_term, tokens):
    aspect_tokens = word_tokenize(aspect_term.lower())
    for i in range(len(tokens) - len(aspect_tokens) + 1):
        if tokens[i:i+len(aspect_tokens)] == aspect_tokens:
            return True
    return False

# Déterminer la polarité d'un aspect
def determine_aspect_polarity(aspect_term, tokens, pos_tags, window=5):
    aspect_tokens = word_tokenize(aspect_term.lower())
    try:
        aspect_index = next(i for i in range(len(tokens) - len(aspect_tokens) + 1) if tokens[i:i+len(aspect_tokens)] == aspect_tokens)
        print(f"Aspect '{aspect_term}' trouvé dans la phrase.")
    except StopIteration:
        print(f"Aspect '{aspect_term}' non trouvé dans la phrase.")
        return "neutral"
    
    start = max(0, aspect_index - window)
    end = min(len(tokens), aspect_index + len(aspect_tokens) + window)
    context = tokens[start:end]
    pos_context = pos_tags[start:end]
    has_negation, negation_index = detect_negation(context)
    polarity = 0
    
    for i, (token, pos) in enumerate(pos_context):
        pos = pos[0].lower()
        if pos in ['n', 'v', 'a', 'r']:
            token_polarity = get_polarity(token, pos)
            if has_negation and i > negation_index:
                token_polarity *= -1
            polarity += token_polarity
    
    if abs(polarity) < 0.1:
        return "neutral"
    elif polarity > 0:
        return "positive"
    else:
        return "negative"

# Évaluation des résultats
# def evaluate_results(y_true, y_pred):
#     if not y_true or not y_pred:
#         print("Erreur : Les listes y_true ou y_pred sont vides. Impossible d'évaluer les résultats.")
#         return
    
#     precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
#     recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
#     f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
#     accuracy = accuracy_score(y_true, y_pred)
    
#     print("\nRésultats d'évaluation :")
#     print(f"Précision: {precision:.2f}, Rappel: {recall:.2f}, F1-score: {f1:.2f}, Accuracy: {accuracy:.2f}")

# Charger les fichiers XML
def load_xml(file_path):
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        print(f"\nLe fichier {file_path} a été chargé avec succès.")
        return root
    except FileNotFoundError:
        print(f"Erreur : Le fichier {file_path} n'existe pas.")
        return None
    except Exception as e:
        print(f"Erreur lors du chargement du fichier {file_path} : {e}")
        return None

# Analyse principale
def analyze_sentiments(train_file, test_file, gold_file):
    print(f"\nChargement des données depuis {train_file}, {test_file}, {gold_file}...")
    train_root = load_xml(train_file)
    test_root = load_xml(test_file)
    gold_root = load_xml(gold_file)
    
    if not test_root or not gold_root:
        print("Erreur : Impossible de charger les fichiers nécessaires.")
        return [], []
    
    y_true = []
    y_pred = []
    
    print("\nAnalyse des sentiments en cours...")
    for sentence in test_root.findall('sentence'):
        text = sentence.find('text').text
        tokens, pos_tags, named_entities = preprocess_sentence(text)
        
        for aspect_term in sentence.findall('aspectTerms/aspectTerm'):
            term = aspect_term.get('term')
            print(f"\nTraitement de l'aspect '{term}'...")
            
            # Prédiction de la polarité de l'aspect
            polarity_pred = determine_aspect_polarity(term, tokens, pos_tags)
            print(f"Polarité prédite pour l'aspect '{term}' : {polarity_pred}")
            
            # Récupération de la polarité gold standard
            gold_polarity = get_gold_polarity(gold_root, sentence.get('id'), term)
            print(f"Polarité gold standard pour l'aspect '{term}' : {gold_polarity}")
            
            # Ajout aux listes de vérité terrain et prédiction
            if gold_polarity and polarity_pred:
                y_true.append(gold_polarity)
                y_pred.append(polarity_pred)
            else:
                print(f"Polarité manquante pour l'aspect '{term}'.")
    
    return y_true, y_pred

# Obtenir la polarité gold standard
def get_gold_polarity(gold_root, sentence_id, aspect_term):
    gold_sentence = gold_root.find(f".//sentence[@id='{sentence_id}']")
    if gold_sentence:
        for gold_aspect_term in gold_sentence.findall(".//aspectTerm"):
            if gold_aspect_term.get("term").lower() == aspect_term.lower():
                return gold_aspect_term.get('polarity')
        print(f"Aspect '{aspect_term}' non trouvé dans le fichier gold standard pour la phrase ID {sentence_id}.")
    else:
        print(f"Sentence ID {sentence_id} non trouvée dans le fichier gold standard.")
    return None

# Chemins des fichiers
restaurants_train = "SemEval'14-ABSA-TrainData_v2 & AnnotationGuidelines/Restaurants_Train.xml"
restaurants_test_no_labels = "SemEval'14-ABSA-TrainData_v2 & AnnotationGuidelines/Restaurants_Test_NoLabels.xml"
restaurants_test_gold = "SemEval'14-ABSA-TrainData_v2 & AnnotationGuidelines/Restaurants_Test_Gold.xml"
laptops_train = "SemEval'14-ABSA-TrainData_v2 & AnnotationGuidelines/Laptop_Train.xml"
laptops_test_no_labels = "SemEval'14-ABSA-TrainData_v2 & AnnotationGuidelines/Laptop_Test_NoLabels.xml"
laptops_test_gold = "SemEval'14-ABSA-TrainData_v2 & AnnotationGuidelines/Laptop_Test_Gold.xml"

# Fonction pour afficher la répartition des polarités
def plot_polarity_distribution(y_true, y_pred, title, ax):
    from collections import Counter
    
    true_counts = Counter(y_true)
    pred_counts = Counter(y_pred)
    
    labels = ['positive', 'negative', 'neutral']
    
    true_values = [true_counts[label] for label in labels]
    pred_values = [pred_counts[label] for label in labels]
    
    x = range(len(labels))
    
    ax.bar([i - 0.2 for i in x], true_values, width=0.4, label='Gold Standard', color='blue')
    ax.bar([i + 0.2 for i in x], pred_values, width=0.4, label='Predictions', color='orange')
    
    ax.set_xlabel('Polarité')
    ax.set_ylabel('Nombre d\'aspects')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

def plot_evaluation_metrics(precision, recall, f1, accuracy, title, ax):
    metrics = ['Précision', 'Rappel', 'F1-Score', 'Exactitude']
    values = [precision, recall, f1, accuracy]
    
    ax.bar(metrics, values, color=['blue', 'green', 'red', 'purple'])
    
    for i, value in enumerate(values):
        ax.text(i, value + 0.01, f"{value:.2f}", ha='center', fontsize=10)
    
    ax.set_title(title)
    ax.set_ylim(0, 1.2)

def evaluate_and_plot(y_true, y_pred, dataset_name):
    if not y_true or not y_pred:
        print("Erreur : Les listes y_true ou y_pred sont vides. Impossible d'évaluer les résultats.")
        return
    
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    
    print(f"\nRésultats pour {dataset_name} :")
    print(f"Précision: {precision:.2f}, Rappel: {recall:.2f}, F1-score: {f1:.2f}, Accuracy: {accuracy:.2f}")
    
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    
    # Afficher la répartition des polarités
    plot_polarity_distribution(y_true, y_pred, f"Répartition des polarités pour {dataset_name}", axs[0])
    
    # Afficher les métriques d'évaluation
    plot_evaluation_metrics(precision, recall, f1, accuracy, f"Métriques d'évaluation pour {dataset_name}", axs[1])
    
    plt.tight_layout()
    plt.show()

# Fonction pour entraîner un modèle combiné
def analyze_combined_sentiments(train_files, test_file, gold_file):
    print(f"\nChargement des données depuis {train_files}, {test_file}, {gold_file}...")
    train_roots = [load_xml(train_file) for train_file in train_files]
    test_root = load_xml(test_file)
    gold_root = load_xml(gold_file)
    
    if not test_root or not gold_root:
        print("Erreur : Impossible de charger les fichiers nécessaires.")
        return [], []
    
    y_true = []
    y_pred = []
    
    print("\nAnalyse des sentiments en cours...")
    for sentence in test_root.findall('sentence'):
        text = sentence.find('text').text
        tokens, pos_tags, named_entities = preprocess_sentence(text)
        
        for aspect_term in sentence.findall('aspectTerms/aspectTerm'):
            term = aspect_term.get('term')
            print(f"\nTraitement de l'aspect '{term}'...")
            
            # Prédiction de la polarité de l'aspect
            polarity_pred = determine_aspect_polarity(term, tokens, pos_tags)
            print(f"Polarité prédite pour l'aspect '{term}' : {polarity_pred}")
            
            # Récupération de la polarité gold standard
            gold_polarity = get_gold_polarity(gold_root, sentence.get('id'), term)
            print(f"Polarité gold standard pour l'aspect '{term}' : {gold_polarity}")
            
            # Ajout aux listes de vérité terrain et prédiction
            if gold_polarity and polarity_pred:
                y_true.append(gold_polarity)
                y_pred.append(polarity_pred)
            else:
                print(f"Polarité manquante pour l'aspect '{term}'.")
    
    return y_true, y_pred

def evaluate_and_plot_polarity(y_true, y_pred, dataset_name):
    if not y_true or not y_pred:
        print("Erreur : Les listes y_true ou y_pred sont vides. Impossible d'évaluer les résultats.")
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Afficher la répartition des polarités
    plot_polarity_distribution(y_true, y_pred, f"Répartition des polarités pour {dataset_name}", ax)
    
    plt.tight_layout()
    plt.show()

def plot_final_comparison(results):
    labels = ["Restaurants", "Ordinateurs Portables", "Restaurants (Ensemble)", "Ordinateurs Portables (Ensemble)"]
    metrics = ["Précision", "Rappel", "F1-Score", "Exactitude"]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, metric in enumerate(metrics):
        values = [float(result[i+1]) for result in results]
        if len(values) != len(labels):
            print(f"Erreur : Les valeurs pour la métrique '{metric}' n'ont pas la même longueur que les labels.")
            return
        ax.plot(labels, values, marker='o', label=metric)
    
    ax.set_xlabel('Approches')
    ax.set_ylabel('Valeurs')
    ax.set_title('Comparaison des Métriques d\'Évaluation')
    ax.legend()
    plt.tight_layout()
    plt.show()

def compare_approaches():
    results = []

    print("Q5 : Résultats pour les restaurants :")
    y_true_restaurants, y_pred_restaurants = analyze_sentiments(restaurants_train, restaurants_test_no_labels, restaurants_test_gold)
    evaluate_and_plot_polarity(y_true_restaurants, y_pred_restaurants, "Restaurants")
    precision_restaurants = precision_score(y_true_restaurants, y_pred_restaurants, average='weighted', zero_division=0)
    recall_restaurants = recall_score(y_true_restaurants, y_pred_restaurants, average='weighted', zero_division=0)
    f1_restaurants = f1_score(y_true_restaurants, y_pred_restaurants, average='weighted', zero_division=0)
    accuracy_restaurants = accuracy_score(y_true_restaurants, y_pred_restaurants)
    results.append(["Restaurants", f"{precision_restaurants:.2f}", f"{recall_restaurants:.2f}", f"{f1_restaurants:.2f}", f"{accuracy_restaurants:.2f}"])

    print("\nQ6 : Résultats pour les ordinateurs portables :")
    y_true_laptops, y_pred_laptops = analyze_sentiments(laptops_train, laptops_test_no_labels, laptops_test_gold)
    evaluate_and_plot_polarity(y_true_laptops, y_pred_laptops, "Ordinateurs Portables")
    precision_laptops = precision_score(y_true_laptops, y_pred_laptops, average='weighted', zero_division=0)
    recall_laptops = recall_score(y_true_laptops, y_pred_laptops, average='weighted', zero_division=0)
    f1_laptops = f1_score(y_true_laptops, y_pred_laptops, average='weighted', zero_division=0)
    accuracy_laptops = accuracy_score(y_true_laptops, y_pred_laptops)
    results.append(["Ordinateurs Portables", f"{precision_laptops:.2f}", f"{recall_laptops:.2f}", f"{f1_laptops:.2f}", f"{accuracy_laptops:.2f}"])

    print("\nQ7 : Résultats pour les deux ensembles (entraînés ensemble) :")
    y_true_combined_train, y_pred_combined_train = analyze_combined_sentiments([restaurants_train, laptops_train], restaurants_test_no_labels, restaurants_test_gold)
    
    # Évaluation pour les restaurants (entraînés ensemble)
    y_true_restaurants_combined = [y for y, t in zip(y_true_combined_train, y_pred_combined_train) if t in y_true_restaurants]
    y_pred_restaurants_combined = [t for y, t in zip(y_true_combined_train, y_pred_combined_train) if t in y_true_restaurants]
    evaluate_and_plot_polarity(y_true_restaurants_combined, y_pred_restaurants_combined, "Restaurants (entraînés ensemble)")
    precision_restaurants_combined = precision_score(y_true_restaurants_combined, y_pred_restaurants_combined, average='weighted', zero_division=0)
    recall_restaurants_combined = recall_score(y_true_restaurants_combined, y_pred_restaurants_combined, average='weighted', zero_division=0)
    f1_restaurants_combined = f1_score(y_true_restaurants_combined, y_pred_restaurants_combined, average='weighted', zero_division=0)
    accuracy_restaurants_combined = accuracy_score(y_true_restaurants_combined, y_pred_restaurants_combined)
    results.append(["Restaurants (Ensemble)", f"{precision_restaurants_combined:.2f}", f"{recall_restaurants_combined:.2f}", f"{f1_restaurants_combined:.2f}", f"{accuracy_restaurants_combined:.2f}"])

    # Évaluation pour les ordinateurs portables (entraînés ensemble)
    y_true_laptops_combined = [y for y, t in zip(y_true_combined_train, y_pred_combined_train) if t in y_true_laptops]
    y_pred_laptops_combined = [t for y, t in zip(y_true_combined_train, y_pred_combined_train) if t in y_true_laptops]
    evaluate_and_plot_polarity(y_true_laptops_combined, y_pred_laptops_combined, "Ordinateurs Portables (entraînés ensemble)")
    precision_laptops_combined = precision_score(y_true_laptops_combined, y_pred_laptops_combined, average='weighted', zero_division=0)
    recall_laptops_combined = recall_score(y_true_laptops_combined, y_pred_laptops_combined, average='weighted', zero_division=0)
    f1_laptops_combined = f1_score(y_true_laptops_combined, y_pred_laptops_combined, average='weighted', zero_division=0)
    accuracy_laptops_combined = accuracy_score(y_true_laptops_combined, y_pred_laptops_combined)
    results.append(["Ordinateurs Portables (Ensemble)", f"{precision_laptops_combined:.2f}", f"{recall_laptops_combined:.2f}", f"{f1_laptops_combined:.2f}", f"{accuracy_laptops_combined:.2f}"])

    # Comparaison globale
    print("\nComparaison globale des quatre approches :")
    plot_final_comparison(results)
    
compare_approaches()