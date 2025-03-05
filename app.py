import json
import os  # Ajout du module os pour parcourir les fichiers
import shutil  # Pour supprimer le répertoire de la base de données
import datetime

from langchain.chains import RetrievalQA

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


"""
Projet Final IA Générative Sorbonne:

Ce Script va permettre de construire un modèle LLM en basant sur 
le RAG (Génération augmentée de récupération) pour améliorer les résultats
de l'IA Générative sur la génération du quiz.

Le but de ce projet est que l'utilisateur va lui demander quels informations précises
pour permettre de générer correctement le contexte du quiz.
L'utilisateur doit définir:
-Le nom du sujet (Aide humanitaire, Crise humanitaire, Droit fondamental, Droit Civil, Droit International)
-L'option des réponses (si les questions possèdent des choix uniques ou multiples)
-Le nombre de tokens possibles pour générer le quiz.
-Le nombre de questions à générer (si possible)

En sortie on obtient ce type de structure du quiz:
- Nom du sujet trouvé dans les documents
- Nouvelle question 
- Une nouvelle liste des options
- Une ou plusieurs nouvelles réponses correcte(s) (Un ou plusieurs réponses)
- Une nouvelle explication (pour comprendre la réponse ou les réponses exactes)
"""



# Fonction pour vérifier si la base de données Chroma existe et la supprimer
def check_and_delete_chroma_db(db_path):
    if os.path.exists(db_path):  # Vérifie si le dossier existe
        print(f"Base de données Chroma trouvée. Suppression du dossier '{db_path}'...")
        shutil.rmtree(db_path)  # Supprime le dossier et tout son contenu
        print("Base de données supprimée.")
    else:
        print("Aucune base de données Chroma trouvée. Création d'une nouvelle base.")


# 1. Chargement des données depuis un dossier de fichiers JSON
def load_data_from_folder(json_folder):
    all_questions = []  # Liste pour stocker toutes les questions
    for filename in os.listdir(json_folder):
        if filename.endswith(".json"):  # Vérifier si le fichier est un JSON
            # Extraire le sujet à partir du nom du fichier
            with open(os.path.join(json_folder, filename), 'r', encoding='utf-8') as f:
                data = json.load(f)
                for question in data.get("questions", []):
                    question["source"] = filename  # Ajouter la source (nom du fichier JSON)
                    all_questions.append(question)  # Ajouter les questions à la liste
    return all_questions

# 2. Transformation des données en documents
def create_documents(all_questions):
    documents = []
    for idx, question in enumerate(all_questions):
        correct_answer = question['reponse_correcte']
        if isinstance(correct_answer, list):
            correct_answer = ", ".join(correct_answer)  # Convertir la liste en chaîne

        metadata = {
            "type": "unique" if "options" in question else "multi",
            "correct_answer": correct_answer,
            "explanation": question['explication'],
            "source": question.get("source", "inconnue"),  # Récupérer la source du fichier JSON
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


# Fonction de récupération avec compression et QA
def retrieve_with_compression_and_qa(vectorstore, query, number_documents, temperature):
    # Description du contenu des documents pour la récupération
    document_content_description = """
    A database of educational quizzes in French on differents topics: 
    Aide humanitaire, Alimentation et nutrition, Crise humanitaire, Droit fondamental, Droit Civil, Droit International. 
    Questions can be unique or multi choice, each with options, correct answers, and a detailed explanation.    
    """

    metadata_field_info = [
        AttributeInfo(
            name="type",
            description="Type options for answser (unique or multi)",
            type="string",
        ),
        AttributeInfo(
            name="source",
            description="The lecture that this chunk is from should be one of the JSON files.",
            type="string",
        ), 
    ]
    
    # Initialisation du modèle de langage
    llm = OpenAI(temperature=temperature, openai_api_key=OPENAI_API_KEY)

    # Récupération via SelfQueryRetriever
    base_retriever = SelfQueryRetriever.from_llm(
        llm,
        vectorstore,
        document_content_description,
        metadata_field_info,
        verbose=True
    )
    
    # Compression des informations avec LLMChainExtractor
    compressor = LLMChainExtractor.from_llm(llm)  # Modèle pour l'extraction
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )

    # Exécution avec compression et récupération QA
    compressed_response = compression_retriever.invoke(query)
    
    # Génération de la réponse avec QA Chain
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": number_documents})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Choisissez un type de chaîne, "stuff" est souvent utilisé pour les Q&A simples
        retriever=retriever,
        verbose=True
    )

    # Combine compression et réponse finale de QA
    final_response = qa_chain.invoke(query)
    return compressed_response, final_response


# Fonction pour générer le quiz
def generate_quiz(retrieved_data, model_name):
    # Log pour voir les données récupérées
    template = """
    A partir de nos documents du JSON génère un quiz éducative en français basé sur ces informations:
    {context}
    Format requis:
    - Nom du sujet demandé
    - Questions variées (choix unique/multi)
    - Options claires
    - Indiquer les réponses correctes
    - Explications concises
    """
    prompt = ChatPromptTemplate.from_template(template)

    # Initialisation de ChatOpenAI avec la clé API
    llm = ChatOpenAI(model=model_name, api_key=OPENAI_API_KEY)

    chain = prompt | llm | StrOutputParser()

    # Ici, on passe directement la chaîne de caractères de `retrieved_data` (contexte)
    context = retrieved_data  # Utilisez la chaîne de caractères renvoyée par RetrievalQA

    return chain.invoke({
        "context": context
    })


def save_history_quiz(quiz,output_folder):
    os.makedirs(output_folder, exist_ok=True)
    # Générer le nom du fichier avec date et heure
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"quiz_{timestamp}.txt"
    file_path = os.path.join(output_folder, file_name)

    # Sauvegarder le quiz dans le fichier
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(quiz)

    print(f"Quiz sauvegardé dans {file_path}")

# Pipeline complet
def main():
    # Configuration
    json_folder = "quiz"  # Remplacez par le chemin de votre dossier JSON
    output_folder = "output_quiz"
    
    #Aide humanitaire, crise humanitaire et droit international marche bien
    query = "droit_civil" # Requête principale pour demander les informations sur notre quiz
    max_number_tokens = 1000  # Nombre max tokens
    number_documents = 3 #Number of best results of documents
    temperature = 0.7  # Température pour déterminer le niveau créativité en sortie
    chroma_db_path = "./chroma_db"  # Le chemin de la base de données Chroma

    # Étape 1: Vérification et suppression de la base de données Chroma existante (si nécessaire)
    check_and_delete_chroma_db(chroma_db_path)

    # Étape 2: Chargement des données depuis le dossier
    all_questions = load_data_from_folder(json_folder)
    
    # Étape 3: Création des documents
    documents = create_documents(all_questions)
    print(f"Nombre total de documents stockés : {len(documents)}")

    # Étape 4: Traitement
    vectorstore = process_documents(documents, max_number_tokens)
    print(f"Nombre de documents dans Chroma: {vectorstore._collection.count()}")

    # Étape 5: Récupération avec compression et QA
    compressed_response, final_response = retrieve_with_compression_and_qa(vectorstore, query,number_documents, temperature)
    print(f"Réponse compressée: {compressed_response}")
    print(f"Réponse finale après QA: {final_response}")

    # Étape 6: Génération du quiz
    quiz = generate_quiz([final_response], model_name="gpt-4-turbo")
    print("Génération du quiz:")
    print(quiz)
    save_history_quiz(quiz,output_folder)

if __name__ == "__main__":
    main()
