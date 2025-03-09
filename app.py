import json
import os
import shutil
import datetime
import re

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


# Fonction pour vérifier si la base de données Chroma existe et la supprimer
def check_and_delete_chroma_db(db_path):
    if os.path.exists(db_path):  # Vérifie si le dossier existe
        print(f"Base de données Chroma trouvée. Suppression du dossier '{db_path}'...")
        shutil.rmtree(db_path)  # Supprime le dossier et tout son contenu
        print("Base de données supprimée.")
    else:
        print("Aucune base de données Chroma trouvée. Création d'une nouvelle base.")


# 1. Chargement des données depuis un fichier JSON unique
def load_data_from_file(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        all_questions = []
        for question in data.get("questions", []):
            question["source"] = json_file  # Ajouter la source (nom du fichier JSON)
            all_questions.append(question)  # Ajouter les questions à la liste
    return all_questions


def extract_subject_from_filename(filename):
    """
    Extrait le sujet du nom du fichier JSON.
    Par exemple, 'droit_fondamental.json' devient 'Droit Fondamental'.
    """
    subject = filename.replace(".json", "")  # Supprimer l'extension .json
    # Remplacer les underscores par des espaces et mettre en majuscule le premier caractère de chaque mot
    subject = subject.replace("_", " ").title()  
    return subject


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
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # Modèle préentrainé 

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
def retrieve_with_compression_and_qa(vectorstore, query, number_documents, temperature, current_topic):
    # Le prompt peut indiquer explicitement que la question doit rester dans le sujet spécifique
    prompt_context = f"""
    Vous êtes un assistant spécialisé dans le domaine du {current_topic}. 
    Lorsque vous générez des réponses, assurez-vous de répondre uniquement sur ce sujet.
    Si la question n'est pas liée à ce sujet, vous pouvez répondre en expliquant que cela ne relève pas de ce domaine.
    Assurer vous que la question soit bien claire et précis.
    """
    
    # Ajouter le contexte à la question
    query_with_context = f"{prompt_context} Question: {query}"
    
    # Utiliser cette version de la question dans votre récupération de données
    document_content_description = f"Documents relatifs au sujet de {current_topic}, assurez-vous de respecter ce domaine."
    metadata_field_info = [
        AttributeInfo(
            name="type",
            description="Type options for answer (unique or multi)",
            type="string",
        ),
        AttributeInfo(
            name="source",
            description="La source des documents",
            type="string",
        ),
    ]
    
    # Initialisation du modèle de langage
    llm = OpenAI(temperature=temperature, openai_api_key=OPENAI_API_KEY)

    # Utilisation d'un SelfQueryRetriever sans filtre explicite de documents
    base_retriever = SelfQueryRetriever.from_llm(
        llm,
        vectorstore,
        document_content_description,
        metadata_field_info,
        verbose=True
    )

    # Compression des informations avec LLMChainExtractor
    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )

    # Récupération et génération de réponse avec QA
    compressed_response = compression_retriever.invoke(query_with_context)
    
    # Génération de la réponse QA
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": number_documents})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        verbose=True
    )

    final_response = qa_chain.invoke(query_with_context)
    return compressed_response, final_response


# Fonction pour générer le quiz
def generate_quiz(retrieved_data, model_name, current_topic):
    # Log pour voir les données récupérées
    template = """
    A partir de nos documents du JSON génère un quiz éducative en français basé sur ces informations:
    {context}
    Sujet: {current_topic}
    Format requis:
    - Nom du sujet demandé par l'utilisateur
    - Nombre de questions si possible demandé par l'utilisateur (Si non spécifié, indiquer "Non défini" et générer le nombre de questions aléatoires)
    - Nombre d'options de réponses si possible demandé par l'utilisateur (Si non spécifié, indiquer "Non défini" et générer le nombre de réponses aléatoires.)
    - Options des réponses (unique ou multi) (Si non spécifié, indiquer "Non défini" et générer les options de réponses aléatoires.)
    - Indiquer les réponses correctes
    - Une explication concise
    """

    # Remplacer `current_topic` et `context` dans le prompt
    prompt = ChatPromptTemplate.from_template(template)

    # Initialisation de ChatOpenAI avec la clé API
    llm = ChatOpenAI(model=model_name, api_key=OPENAI_API_KEY)

    chain = prompt | llm | StrOutputParser()

    # Passer à la fois le contexte et la question dans les variables
    context = retrieved_data  # Utilisez la chaîne de caractères renvoyée par RetrievalQA
    query = f"Génère un quiz sur le sujet : {current_topic}"  # Assurez-vous de passer la question correctement

    return chain.invoke({
        "context": context,
        "current_topic": current_topic,
        "query": query  
    })


def save_history_quiz(quiz, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    # Générer le nom du fichier avec date et heure
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"quiz_{timestamp}.txt"
    file_path = os.path.join(output_folder, file_name)

    # Sauvegarder le quiz dans le fichier
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(quiz)

    print(f"Quiz sauvegardé dans {file_path}")

def handle_no_answer(final_response):
    # Motifs à rechercher dans la réponse (ajout de plusieurs variantes en français et en anglais)
    patterns = [
        r"je ne peux pas",            
        r"je ne sais pas",            
        r"je ne peux pas répondre",   
        r"i can't",                   
        r"i don't know",              
        r"i am unable",               
        r"i have no information",     
        r"sorry",                     
        r"sorry, i can't",            
        r"sorry, i don't know",       
        r"sorry, i am unable",       
        r"désolé",                    
        r"je n'ai pas",               
        r"je ne suis pas sûr",        
        r"je ne suis pas certaine",  
        r"je n'ai aucune information",  
        r"je ne sais pas répondre",   
        r"je ne peux pas vous aider", 
        r"ce n'est pas mon domaine",  
        r"cela ne relève pas de ce domaine", 
        r"je ne suis pas en mesure de répondre",  
        r"ce n'est pas dans mon domaine",        
        r"je ne peux pas répondre à cette question" 
    ]

    for pattern in patterns:
        if re.search(pattern, final_response["result"].strip().lower()):
            print("Je n'ai pas compris votre question. Veuillez reformuler.")
            return True
    
    return False


def main():
    # Variables de configuration
    json_folder = "quiz"
    json_file = f"{json_folder}/droit_international.json"  # Remplacer par le nom du fichier JSON
    output_folder = "output_quiz"

    max_number_tokens = 1000
    number_documents = 3
    temperature = 0.7
    chroma_db_path = "./chroma_db"

    """
    Requête de l'utilisateur pour demander les informations précises 
    par exemple (nombre de questions possibles, réponses uniques ou multi, nombre de réponses" 
    Si le sujet est hors sujet il va déterminer automatiquement le topic dans la base de données qui a déjà choisi dans la liste attribuée.
    Format requis:
    - Nom du sujet demandé par l'utilisateur
    - Nombre de questions si possible demandé par l'utilisateur 
    - Nombre d'options de réponses si possible demandé par l'utilisateur 
    - Options des réponses (unique ou multi) 
    - Indiquer les réponses correctes
    - Une explication concise
    """
    query = "Génére moi 5 questions précisant sur la peine de mort"  
    
    # Spécifier le nom du fichier JSON
    json_file = f"{json_folder}/droit_fondamental.json"  # Remplacer par le nom du fichier JSON

    # Extraire le sujet à partir du nom du fichier
    current_topic = extract_subject_from_filename(os.path.basename(json_file))
    print(f"Sujet extrait : {current_topic}")

    # Vérifier si le fichier existe
    if not os.path.exists(json_file):
        print(f"Le fichier '{json_file}' n'existe pas.")
        return
    
    # Étape 1: Vérification et suppression de la base de données Chroma existante
    check_and_delete_chroma_db(chroma_db_path)
    
    # Étape 2: Chargement des données depuis le fichier JSON
    all_questions = load_data_from_file(json_file)
    
    # Étape 3: Création des documents
    documents = create_documents(all_questions)
    
    # Étape 4: Traitement des documents et création du vecteur store
    vectorstore = process_documents(documents, max_number_tokens)
    
    # Étape 6: Récupération avec compression et QA
    compression_response , final_response = retrieve_with_compression_and_qa(vectorstore, query, number_documents, temperature, current_topic)
    print("compression_response", compression_response)
    print("final_response", final_response)

    if handle_no_answer(final_response):
        return

    # Étape 7: Génération du quiz
    quiz = generate_quiz([final_response], model_name="gpt-4-turbo", current_topic=current_topic)
    
    print("Quiz généré :")
    print(quiz)
    save_history_quiz(quiz, output_folder)


if __name__ == "__main__":
    main()
