import json
import os

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


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
def transform_documents(all_questions):
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
def split_documents_embedding(documents,chroma_path,max_tokens=1000):
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
        persist_directory=chroma_path
    )
    
    # Ajout des documents dans Chroma
    vectorstore.add_texts(texts=split_texts, metadatas=split_metadatas, ids=split_ids)
    
    return vectorstore


# Fonction de récupération avec compression et QA
def retrieve_qa(vectorstore, query, number_documents, temperature, current_topic,model_name):
    # Le prompt peut indiquer explicitement que la question doit rester dans le sujet spécifique
    prompt_context = f"""
    Vous êtes un assistant spécialisé dans le domaine du {current_topic}. 
    L'utilisateur vous demande si possible de générer le quiz sur les documents avec {query}. Le quiz doit contenir :
    - Les informations que vous voulez rechercher sur notre quiz (questions, réponses et explications)
    - Un nombre spécifique de questions,
    - Des options de réponses spécifiques.
    - Indication un ou plusieurs réponses correctes
    - Explication pour chaque réponse
    """

    query_with_context = f"{prompt_context} \n\nGénère un quiz sur {current_topic} avec les détails suivants : {query}"

    # Initialisation du modèle de langage
    llm = ChatOpenAI(model=model_name,temperature=temperature, openai_api_key=OPENAI_API_KEY)

    # Génération de la réponse QA
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": number_documents})

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm= llm,
        retriever=retriever,
        memory=memory,
        verbose=True
    )

    final_response = qa_chain.invoke({"question": query_with_context})

    return final_response