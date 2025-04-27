import os
from weaviate.connect import ConnectionParams
from weaviate import WeaviateClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_weaviate import WeaviateVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langserve import add_routes
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import warnings
import time
from threading import Lock
import logging

# Configuration des logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration du rate limiting
RATE_LIMIT = 3  # Nombre maximum d'appels par seconde
RATE_WINDOW = 1  # Fenêtre de temps en secondes
last_request_time = 0
request_lock = Lock()

def wait_for_rate_limit():
    """Attend si nécessaire pour respecter la limite de taux"""
    global last_request_time
    with request_lock:
        current_time = time.time()
        time_since_last_request = current_time - last_request_time
        if time_since_last_request < RATE_WINDOW / RATE_LIMIT:
            sleep_time = (RATE_WINDOW / RATE_LIMIT) - time_since_last_request
            time.sleep(sleep_time)
        last_request_time = time.time()

class RateLimitedChatOpenAI(ChatOpenAI):
    def _call(self, *args, **kwargs):
        wait_for_rate_limit()
        return super()._call(*args, **kwargs)

def setup_weaviate():
    try:
        logger.info("Connexion à Weaviate...")
        client = WeaviateClient(
            connection_params=ConnectionParams.from_url(
                os.getenv("WEAVIATE_URL", "http://15.237.74.195:8080"),
                50051
            )
        )
        client.connect()
        logger.info("Connexion à Weaviate établie avec succès")
        return client
    except Exception as e:
        logger.error(f"Erreur de connexion à Weaviate: {str(e)}")
        raise e

def create_custom_prompt():
    template = """Tu es un assistant viticole expert en pulvérisateurs et produits phytosanitaires. 
Voici les documents techniques pertinents pour répondre à la question :

{context}

Question : {question}

Instructions :
1. Réponds de manière factuelle uniquement en te basant sur les documents fournis
2. Si l'information n'est pas dans les documents, dis-le clairement
3. Sois précis sur les caractéristiques techniques
4. Cite les sources des informations que tu utilises
5. Si tu cites un document, mentionne son type (PDF, CSV, TXT) et sa position dans le document (chunk X sur Y)
6. Pour les produits phytosanitaires, mentionne toujours :
   - Le nom du produit
   - La catégorie (bio, conventionnel, etc.)
   - Les maladies/ravageurs ciblés
   - Les précautions d'utilisation

Réponse :"""
    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

def setup_rag_chain():
    try:
        logger.info("Initialisation de la chaîne RAG...")
        client = setup_weaviate()
        
        logger.info("Configuration des embeddings...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        logger.info("Configuration du vectorstore...")
        vectorstore = WeaviateVectorStore(
            client=client,
            embedding=embeddings,
            index_name="Document",
            text_key="text"
        )
        
        logger.info("Configuration du LLM...")
        llm = RateLimitedChatOpenAI(
            openai_api_key=os.getenv("key_openai"),
            model_name="gpt-4-turbo-preview",
            temperature=0,
            streaming=True
        )
        
        logger.info("Configuration de la mémoire...")
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        logger.info("Configuration du retriever...")
        base_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 30}
        )

        # Re-ranking avec le LLM
        compressor = LLMChainExtractor.from_llm(llm)
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )

        logger.info("Création de la chaîne de conversation...")
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            combine_docs_chain_kwargs={"prompt": create_custom_prompt()},
            return_source_documents=True,
            return_generated_question=True,
            output_key="answer"
        )
        
        logger.info("Chaîne RAG créée avec succès")
        return chain
    except Exception as e:
        logger.error(f"Erreur lors de la création de la chaîne RAG: {str(e)}")
        raise e

# Initialisation de l'app FastAPI
app = FastAPI(
    title="Vitizen Chat API",
    description="API de chat pour l'assistant viticole",
    version="1.0.0"
)

# Page d'accueil
@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
        <head>
            <title>Vitizen Chat API</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                    line-height: 1.6;
                }
                h1 {
                    color: #2c3e50;
                }
                .endpoint {
                    background-color: #f8f9fa;
                    padding: 15px;
                    border-radius: 5px;
                    margin: 10px 0;
                }
                .endpoint h3 {
                    margin-top: 0;
                    color: #3498db;
                }
                code {
                    background-color: #e9ecef;
                    padding: 2px 5px;
                    border-radius: 3px;
                }
            </style>
        </head>
        <body>
            <h1>Bienvenue sur l'API Vitizen Chat</h1>
            <p>Cette API permet d'interagir avec un assistant viticole expert en pulvérisateurs et produits phytosanitaires.</p>
            
            <div class="endpoint">
                <h3>Endpoints disponibles :</h3>
                <ul>
                    <li><code>/chat</code> - Endpoint principal pour les requêtes de chat</li>
                    <li><code>/chat/playground</code> - Interface de test interactive</li>
                    <li><code>/docs</code> - Documentation Swagger de l'API</li>
                    <li><code>/health</code> - Vérification de l'état du serveur</li>
                </ul>
            </div>
            
            <div class="endpoint">
                <h3>Exemples de questions :</h3>
                <ul>
                    <li>Quelles sont les caractéristiques techniques des pulvérisateurs ?</li>
                    <li>Comment fonctionne un pulvérisateur face par face ?</li>
                    <li>Quelles sont les différentes technologies de pulvérisation ?</li>
                    <li>Quels sont les produits phytosanitaires disponibles ?</li>
                </ul>
            </div>
            
            <p>Pour commencer, visitez <a href="/chat/playground">le playground</a> ou consultez la <a href="/docs">documentation</a>.</p>
        </body>
    </html>
    """

try:
    # Création de la chaîne RAG
    logger.info("Initialisation de la chaîne RAG...")
    rag_chain = setup_rag_chain()
    logger.info("Chaîne RAG initialisée avec succès")

    # Configuration de LangServe sur l'app FastAPI
    logger.info("Configuration de LangServe...")
    add_routes(
        app,
        rag_chain,
        path="/chat"
    )
    logger.info("LangServe configuré avec succès")

except Exception as e:
    logger.error(f"Erreur lors de l'initialisation: {str(e)}")
    raise e

# Ajout d'un endpoint healthcheck
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# Lancement du serveur si exécuté directement
if __name__ == "__main__":
    logger.info("Démarrage du serveur...")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
