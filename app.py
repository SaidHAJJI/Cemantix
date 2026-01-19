import streamlit as st
import os
import chromadb
import google.generativeai as genai
from sentence_transformers import SentenceTransformer, CrossEncoder

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Wiki RAG Explorer", page_icon="📚", layout="wide")

# --- CHARGEMENT DES MODÈLES (EN CACHE) ---
@st.cache_resource
def load_models():
    embed_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    # Configuration Gemini
    api_key = os.getenv("GEMINI_API_KEY")
    genai.configure(api_key=api_key)
    llm = genai.GenerativeModel('gemini-1.5-flash')
    return embed_model, reranker, llm

embed_model, reranker, llm = load_models()

# --- CONNEXION CHROMADB ---
client = chromadb.PersistentClient(path="./chroma_db_wiki")
collection = client.get_or_create_collection(name="wiki_fr_expert")

# --- INTERFACE UTILISATEUR ---
st.title("📚 Assistant Expert Wikipédia (RAG + Rerank)")
st.markdown("Posez une question sur les documents indexés dans votre base locale.")

with st.sidebar:
    st.header("Paramètres")
    n_results = st.slider("Nombre de documents à extraire", 5, 20, 10)
    st.info(f"Documents en base : {collection.count()}")
    if st.button("Ré-indexer (Reset)"):
        # Logique pour vider/remplir la base si besoin
        pass

# --- ZONE DE RECHERCHE ---
question = st.text_input("Votre question :", placeholder="Ex: Qui a fondé Carthage ?")

if question:
    with st.spinner("Recherche et analyse en cours..."):
        # 1. Retrieval
        results = collection.query(query_texts=[question], n_results=n_results)
        
        # 2. Reranking
        candidats = results['documents'][0]
        metas = results['metadatas'][0]
        pairs = [[question, c] for c in candidats]
        scores = reranker.predict(pairs)
        
        scored = sorted(zip(scores, candidats, metas), key=lambda x: x[0], reverse=True)
        top_score, top_text, top_meta = scored[0]

        # 3. Génération Gemini
        prompt = f"Réponds à la question en utilisant ce contexte : {top_text}\n\nQuestion : {question}"
        response = llm.generate_content(prompt)

    # --- AFFICHAGE DES RÉSULTATS ---
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Réponse de l'IA")
        st.write(response.text)
        
    with col2:
        st.subheader("Source la plus proche")
        st.caption(f"Titre : {top_meta['title']}")
        st.caption(f"Score Rerank : {top_score:.2f}")
        with st.expander("Voir le texte source"):
            st.write(top_text)

    # Affichage de tous les candidats pour comparer
    with st.expander("Classement de tous les documents extraits"):
        for i, (s, t, m) in enumerate(scored):
            st.write(f"**{i+1}. {m['title']}** (Score: {s:.2f})")