import os
import io
import datetime
from rag import *
import streamlit as st


#Interface Streamlit
st.title("Générateur de format de Quiz personnalisé avec IA")

if "history" not in st.session_state:
    st.session_state.history = []

# Sélection du fichier JSON
dossier_json = "load_documents"
model_name = "gpt-3.5-turbo"
chroma_path = "./chroma_db"


st.subheader("🎯 Générer un Quiz")
fichiers_disponibles = [f for f in os.listdir(dossier_json) if f.endswith(".json")]
selected_file = st.selectbox("Sélectionnez un sujet", fichiers_disponibles)
temperature = st.slider("Selectionner la température:", 0.1, 1.0, 0.7)
nb_max_tokens = st.slider("Insérer le maximum de tokens", 500,2000, 1000)
nb_documents = st.number_input("Insérer le nombre de documents",value=30,placeholder="Type a number...")

# Zone de texte pour la requête de l'utilisateur
information = st.markdown(
    """
    Saisissez les informations suivants pour rechercher les éléments du quiz :
    Vous pouvez par exemple définir: 
    - Le nombre spécifique de questions
    - Le nombre d'options de réponses spécifiques
    - Avec ou sans indication des réponses correctes
    - Avec ou sans explication pour chaque réponse
    - Indiquer un ou plusieurs réponses correctes 
    """)
example = st.markdown("Exemple d'une requête: 5 questions, 3 options, avec indication de réponses et sans explication")

user_query = st.text_area("Entrez votre requête ici")

# Bouton pour générer le quiz
if st.button("Générer le Quiz"):
    json_file_path = os.path.join(dossier_json, selected_file)
    current_topic = extract_subject_from_filename(selected_file)
    questions = load_data_from_file(json_file_path)
    documents = transform_documents(questions)
    print("Nombre de documents",len(documents))
    vectorstore = split_documents_embedding(documents, chroma_path,nb_max_tokens)
    retrieved_data = retrieve_qa(vectorstore, user_query, nb_documents, temperature, current_topic,model_name)
    quiz = retrieved_data["answer"]
    quiz_lines = quiz.split("\n")
    st.subheader("Quiz généré")
    for line in quiz_lines:
        st.write(line)
        
    quiz_text = "\n".join(quiz_lines)
    quiz_bytes = io.BytesIO(quiz_text.encode("utf-8"))

    # Ajouter un bouton de téléchargement
    st.download_button(
        label="📥 Télécharger le Quiz",
        data=quiz_bytes,
        file_name=f"quiz_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt",
        mime="text/plain"
    )

    st.session_state.history.append({
        "topic": current_topic,
        "query": user_query,
        "quiz": quiz
    })

# 📌 Barre latérale : Historique des quiz générés
st.sidebar.title("📌 Historique des Quiz")

if st.session_state.history:
    with st.sidebar.expander("📜 Voir l'historique"):
        for i, entry in enumerate((st.session_state.history), start=1):
            st.sidebar.markdown(f"### 🔹 Requête {i} - {entry['topic']}")
            st.sidebar.markdown(f"**📝 Requête :** {entry['query']}")
            show_quiz = st.sidebar.checkbox(f"📖 Voir le quiz {i}", key=f"quiz_{i}")

            if show_quiz:
                st.sidebar.code(entry['quiz'], language="text")
else:
    st.sidebar.write("Aucun historique disponible.")

# Bouton pour réinitialiser l'historique
if st.sidebar.button("🗑 Effacer l'historique"):
    st.session_state.history = []
    st.sidebar.success("Historique effacé avec succès !")