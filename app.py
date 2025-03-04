import json
import os  # Ajout du module os pour parcourir les fichiers

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from dotenv import load_dotenv


# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Liste des sujets possibles en fonction du fichier JSON
subject_mapping = {
    "aide_humanitaire": "Aide Humanitaire",
    "alimentation_et_nutrition": "Alimentation et Nutrition",
    "conflits": "Conflits",
    "crise_humanitaire": "Crise Humanitaire",
    "cyberharcelement": "Cyberharcèlement",
    "droit_civil": "Droit Civil",
    "droit_fondamental": "Droit Fondamental",
    "droit_international": "Droit International",
    "esport_et_jeux_video": "Esport et Jeux Vidéo",
    "inegalites": "Inégalités",
    "intelligence_artificielle": "Intelligence Artificielle",
    "mouvements_sociaux": "Mouvements Sociaux",
    "rechauffement_climatique": "Réchauffement Climatique",
}

# 1. Chargement des données depuis un dossier de fichiers JSON
def load_data_from_folder(json_folder):
    all_questions = []  # Liste pour stocker toutes les questions
    for filename in os.listdir(json_folder):
        if filename.endswith(".json"):  # Vérifier si le fichier est un JSON
            # Extraire le sujet à partir du nom du fichier
            subject_name = filename.replace(".json", "").lower().replace(" ", "_")
            subject = subject_mapping.get(subject_name, "General")  # Si le sujet n'est pas trouvé, utiliser "General"

            with open(os.path.join(json_folder, filename), 'r', encoding='utf-8') as f:
                data = json.load(f)
                for question in data.get("questions", []):
                    question["sujet"] = subject  # Ajouter le sujet à chaque question
                    all_questions.append(question)  # Ajouter les questions à la liste
    return all_questions

# 2. Transformation des données en documents
def create_documents(all_questions):
    documents = []
    for idx, question in enumerate(all_questions):
        sujet = question.get('sujet', '')  
        # Convertir la réponse correcte en chaîne de caractères si c'est une liste
        correct_answer = question['reponse_correcte']
        if isinstance(correct_answer, list):
            correct_answer = ", ".join(correct_answer)  # Convertir la liste en chaîne séparée par des virgules
        
        metadata = {
            "type": "single" if "options" in question else "multi",
            "correct_answer": correct_answer,
            "explanation": question['explication'],
            "topic": sujet if sujet else "general",  # Assurez-vous que 'sujet' n'est pas vide
        }
        content = f"Question: {question['question']}\n"
        
        if "options" in question:
            for opt, text in question['options'].items():
                content += f"{opt}: {text}\n"
        else:
            for opt, text in question['multi_options'].items():
                content += f"{opt}: {text}\n"
        
        documents.append((f"doc_{idx}", content, metadata))
    return documents

# 3. Découpage et embedding
def process_documents(documents, max_tokens=1000):
    # Initialiser le text_splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=max_tokens, chunk_overlap=200)
    
    # Préparer les listes pour les textes, métadonnées et IDs
    split_texts = []
    split_metadatas = []
    split_ids = []

    # Parcourir chaque document
    for doc_id, content, metadata in documents:
        # Diviser le texte en morceaux
        chunks = text_splitter.split_text(content)
        
        # Ajouter chaque morceau avec ses métadonnées et un ID unique
        for i, chunk in enumerate(chunks):
            split_texts.append(chunk)
            split_metadatas.append(metadata)  # Les mêmes métadonnées pour chaque morceau
            split_ids.append(f"{doc_id}_chunk_{i}")  # ID unique pour chaque morceau
    
    # Embedding avec Sentence Transformers
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # Modèle préentrainé basé sur

    # Création du vecteur store avec Chroma (compatible LangChain)
    vectorstore = Chroma(
        collection_name="quiz_collection",
        embedding_function=embedding_function,
        persist_directory="./chroma_db"
    )
    
    # Ajout des documents dans Chroma
    vectorstore.add_texts(texts=split_texts, metadatas=split_metadatas, ids=split_ids)
    
    return vectorstore

# Fonction de récupération et génération du quiz
def retrieve_with_compression(vectorstore, temperature, query, k=5):
    document_content_description = "Quiz Education Pédagogique"

    metadata_field_info = [
        AttributeInfo(
            name="type",
            description="Le type de question (single ou multi)",
            type="string",
        ),
        AttributeInfo(
            name="topic",
            description="Le sujet de la question (aide humanitaire, droit juridique, vulgarisation et social)",
            type="string",
        ),
    ]
    
    llm = OpenAI(temperature=temperature, openai_api_key=OPENAI_API_KEY)
    base_retriever = SelfQueryRetriever.from_llm(
        llm,
        vectorstore,
        document_content_description,
        metadata_field_info,
        verbose=True
    )
    
    # 2. Compression Contextuelle
    compressor = LLMChainExtractor.from_llm(ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY))  # Modèle pour l'extraction
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )

    raw_results = base_retriever.invoke(query)
    print(f"Documents retournés par SelfQueryRetriever pour la requête '{query}': {raw_results}")
    for doc in raw_results:
        print(f"Doc ID: {doc.id}, Sujet: {doc.metadata['topic']}, Contenu: {doc.page_content}")
    print('---------------------------------------------------------')
    print('---------------------------------------------------------')
    print('---------------------------------------------------------')
    print('---------------------------------------------------------')
    # Exécuter avec compression
    return compression_retriever.invoke(query)

# Fonction pour générer le quiz
def generate_quiz(retrieved_data, model_name="gpt-3.5-turbo"):
    # Log pour voir les données récupérées
    template = """
    Génère un quiz éducative en français basé sur ces informations:
    {context}
    
    Format requis:
    - Questions variées (choix unique/multiple)
    - Options claires
    - Indiquer les réponses correctes
    - Explications concises
    """
    prompt = ChatPromptTemplate.from_template(template)

    # Initialisation de ChatOpenAI avec la clé API
    llm = ChatOpenAI(model=model_name, api_key=OPENAI_API_KEY)

    chain = prompt | llm | StrOutputParser()

    context = "\n\n".join([doc.page_content for doc in retrieved_data])

    return chain.invoke({
        "context": context
    })

# Pipeline complet
def main():
    # Configuration
    JSON_FOLDER = "quiz"  # Remplacez par le chemin de votre dossier JSON
    
    # Étape 1: Chargement des données depuis le dossier
    all_questions = load_data_from_folder(JSON_FOLDER)
    
    # Étape 2: Création des documents
    documents = create_documents(all_questions)
    
    # Nombre de tokens à définir
    max_number_tokens = 2000
    temperature = 1

    # Étape 3: Traitement
    collection = process_documents(documents, max_number_tokens)
    print(f"Nombre de documents dans Chroma: {collection._collection.count()}")
    
    # Étape 4: Récupération
    query = "intelligence artificielle"
    results = retrieve_with_compression(collection, temperature, query, k=3)
    print(f"Résultats avec requête simple: {results}")
    print('---------------------------------------------------------')
    print('---------------------------------------------------------')
    print('---------------------------------------------------------')
    print('---------------------------------------------------------')
    # Étape 5: Génération
    quiz = generate_quiz(results)
    print("Génération du quiz:")
    print(quiz)

if __name__ == "__main__":
    main()
