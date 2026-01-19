import streamlit as st
from datasets import load_dataset
import os
import chromadb
import google.generativeai as genai
from sentence_transformers import SentenceTransformer, CrossEncoder

def remplir_base():
    with st.spinner("Initialisation de la base depuis le fichier local..."):
        try:
            # On crée manuellement quelques documents de test si le fichier n'existe pas
            docs = [
                "Carthage, située dans l'actuelle Tunisie, fut fondée par les Phéniciens en 814 av. J.-C.",
                "L'intelligence artificielle repose sur des réseaux de neurones profonds et des transformeurs.",
                "Le RAG (Retrieval Augmented Generation) permet d'optimiser les réponses des LLM.",
                "La Tunisie possède un riche patrimoine historique allant de l'époque punique à l'époque romaine."
            ]
            
            ids = [f"id_{i}" for i in range(len(docs))]
            metas = [{"title": "Document de test"} for _ in range(len(docs))]
            
            collection.add(ids=ids, documents=docs, metadatas=metas)
            st.success(f"Base initialisée avec {len(docs)} documents de secours !")
            st.rerun()
                
        except Exception as e:
            st.error(f"Erreur : {str(e)}")

# --- CONNEXION CHROMADB ---
client = chromadb.PersistentClient(path="./chroma_db_wiki")
collection = client.get_or_create_collection(name="wiki_fr_expert")

# --- GESTION DU BOUTON RESET ---
with st.sidebar:
    st.header("Paramètres de recherche")
    # On limite le max au nombre de documents réellement présents (minimum 1)
    max_docs = max(1, collection.count())
    n_results = st.slider("Nombre de documents à extraire", 1, min(max_docs, 10), min(max_docs, 3))
    # ------------------

    if st.button("Ré-indexer (Reset)"):
        client.delete_collection(name="wiki_fr_expert")
        collection = client.create_collection(name="wiki_fr_expert")
        remplir_base()
        st.rerun()

    count = collection.count()
    st.info(f"Documents en base : {count}")
    
    if count == 0:
        if st.button("Initialiser la base"):
            remplir_base()
            st.rerun()

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
    # Au lieu de :
# llm = genai.GenerativeModel('gemini-1.5-flash')

# Essayez le nom standard :
    llm = genai.GenerativeModel('gemini-1.5-flash-latest')
    return embed_model, reranker, llm

embed_model, reranker, llm = load_models()

# --- INTERFACE UTILISATEUR ---
st.title("📚 Assistant Expert Wikipédia (RAG + Rerank)")
st.markdown("Posez une question sur les documents indexés dans votre base locale.")

# --- ZONE DE RECHERCHE ---
question = st.text_input("Votre question :", placeholder="Ex: Qui a fondé Carthage ?")

if question:
    with st.spinner("Recherche et analyse en cours..."):
        # 1. Retrieval (On utilise n_results défini dans la sidebar)
        results = collection.query(query_texts=[question], n_results=n_results)
        
        # Vérification de sécurité pour éviter l'erreur "index out of range"
        if not results['documents'] or len(results['documents'][0]) == 0:
            st.warning("Désolé, aucun document dans la base ne semble correspondre à votre question.")
        else:
            # 2. Reranking
            candidats = results['documents'][0]
            metas = results['metadatas'][0]
            
            # ... suite de votre code (CrossEncoder, Gemini, etc.)
        pairs = [[question, c] for c in candidats]
        scores = reranker.predict(pairs)
        
        scored = sorted(zip(scores, candidats, metas), key=lambda x: x[0], reverse=True)
        # ... après le calcul de 'scored'
        if len(scored) > 0:
            top_score, top_text, top_meta = scored[0]
            # ... reste du code de génération
        else:
            st.error("Aucun document pertinent n'a été trouvé dans la base de données.")
            st.stop() # Arrête le script proprement pour cette exécution
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