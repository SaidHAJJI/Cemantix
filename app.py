import streamlit as st
from datasets import load_dataset
import os
import chromadb
import google.generativeai as genai
from sentence_transformers import SentenceTransformer, CrossEncoder

def remplir_base():
    with st.spinner("Chargement de Wikipédia (Format Parquet sécurisé)..."):
        try:
            # On utilise une source moderne sans script Python
            dataset = load_dataset(
                "graelo/wikipedia", 
                "20231101.fr", 
                split="train", 
                streaming=True
            )
            
            docs, ids, metas = [], [], []
            for i, entry in enumerate(dataset):
                if i >= 150: break 
                
                # Extraction propre du contenu
                content = entry.get("text") or ""
                title = entry.get("title") or "Sans titre"
                
                if content:
                    docs.append(content[:1000]) # On limite la taille pour le test
                    ids.append(f"id_{i}")
                    metas.append({"title": title})
            
            if docs:
                collection.add(ids=ids, documents=docs, metadatas=metas)
                st.success(f"Base initialisée avec {len(docs)} articles !")
                st.rerun()
            else:
                st.error("Aucune donnée extraite du dataset.")
                
        except Exception as e:
            st.error(f"Erreur critique : {str(e)}")
            st.info("Astuce : Vérifiez que 'datasets>=2.16.0' est dans requirements.txt")

# --- CONNEXION CHROMADB ---
client = chromadb.PersistentClient(path="./chroma_db_wiki")
collection = client.get_or_create_collection(name="wiki_fr_expert")

# --- GESTION DU BOUTON RESET ---
with st.sidebar:
    st.header("Gestion des données")
    if st.button("Ré-indexer (Reset)"):
        # On supprime et on recrée la collection
        client.delete_collection(name="wiki_fr_expert")
        collection = client.create_collection(name="wiki_fr_expert")
        remplir_base()
        st.rerun() # Relance l'application pour mettre à jour l'affichage

    # Vérification automatique si vide
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
    llm = genai.GenerativeModel('gemini-1.5-flash')
    return embed_model, reranker, llm

embed_model, reranker, llm = load_models()

# --- INTERFACE UTILISATEUR ---
st.title("📚 Assistant Expert Wikipédia (RAG + Rerank)")
st.markdown("Posez une question sur les documents indexés dans votre base locale.")

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