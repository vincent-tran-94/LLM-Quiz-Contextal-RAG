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


OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# 1. Chargement des données depuis un dossier de fichiers JSON
def load_data_from_folder(json_folder):
    all_questions = []  # Liste pour stocker toutes les questions
    for filename in os.listdir(json_folder):
        if filename.endswith(".json"):  # Vérifier si le fichier est un JSON
            with open(os.path.join(json_folder, filename), 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_questions.extend(data["questions"])  # Ajouter les questions à la liste
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
def process_documents(documents,max_tokens=1000):
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
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") #Modèle préentrainé basé sur

    # Création du vecteur store avec Chroma (compatible LangChain)
    vectorstore = Chroma(
        collection_name="quiz_collection",
        embedding_function=embedding_function,
        persist_directory="./chroma_db"
    )
    
    # Ajout des documents dans Chroma
    vectorstore.add_texts(texts=split_texts, metadatas=split_metadatas, ids=split_ids)
    
    return vectorstore


def retrieve_with_compression(vectorstore,temperature, query, k=5):
    document_content_description = "Quiz Education Pédagogique"

    metadata_field_info = [
        AttributeInfo(
            name="type",
            description="Le type de question (single ou multi)",
            type="string",
        ),
        AttributeInfo(
            name="topic",
            description="Le sujet de la question (humanitaire, droit juridique, vulgarisation et social)",
            type="string",
        ),
    ]
    
    llm = OpenAI(temperature=temperature,openai_api_key=OPENAI_API_KEY)
    base_retriever = SelfQueryRetriever.from_llm(
        llm,
        vectorstore,
        document_content_description,
        metadata_field_info,
        verbose=True
    )
    
    # 2. Compression Contextuelle
    compressor = LLMChainExtractor.from_llm(ChatOpenAI(temperature=0,openai_api_key=OPENAI_API_KEY))  # Modèle pour l'extraction
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )

    raw_results = base_retriever.invoke(query)
    print(f"Documents retournés par SelfQueryRetriever: {raw_results}")
    
    # Exécuter avec compression
    return compression_retriever.invoke(query)[:k]


def generate_quiz(retrieved_data, model_name="gpt-3.5-turbo"):
    print(f"Type de retrieved_data: {type(retrieved_data)}")
    print(f"Contenu de retrieved_data: {retrieved_data}")
    if retrieved_data:
        print(f"Premier élément: {retrieved_data[0]}")

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
    
    #Nombre de tokens à définir
    number_tokens = 1000
    temperature = 1

    # Étape 3: Traitement
    collection = process_documents(documents,number_tokens)
    print(f"Nombre de documents dans Chroma: {collection._collection.count()}")
    
    # # Étape 4: Récupération
    query = "aide sociale"
    results = retrieve_with_compression(collection,temperature,query, k=3)
    print(f"Résultats avec requête simple: {results}")

    # Étape 5: Génération
    quiz = generate_quiz(results)
    print(quiz)

if __name__ == "__main__":
    main()