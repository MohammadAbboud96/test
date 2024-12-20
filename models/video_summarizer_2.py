from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Milvus, Chroma
from langchain.chains import ConversationalRetrievalChain
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import CharacterTextSplitter
import os
from chromadb.config import Settings
from langchain.embeddings.base import Embeddings

class VideoSummarizer:
    def __init__(self, vector_db="milvus"):
        self.chat_model = ChatOpenAI(temperature=0, model_name="gpt-4")
        self.embeddings = OpenAIEmbeddings()

        class EmbeddingFunction(Embeddings):
            def embed_documents(self, texts):
                return OpenAIEmbeddings().embed_documents(texts)

            def embed_query(self, text):
                return OpenAIEmbeddings().embed_query(text)

        if vector_db == "milvus":
            self.db = Milvus(
                embedding_function=self.embeddings,
                connection_args={"host": "localhost", "port": 19530},
                collection_name="video_summaries",
            )
        elif vector_db == "chroma":
            self.db = Chroma(
                embedding_function=self.embeddings,
                persist_directory="./chroma_data",
                client_settings=Settings(anonymized_telemetry= False)
            )
        else:
            raise ValueError("Unsupported vector database. Use 'milvus' or 'chroma'.")

        self.text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    def fetch_transcript(self, video_url):
        video_id = video_url.split("v=")[-1]
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([segment['text'] for segment in transcript])

    def summarize_transcript(self, transcript):
        prompt = f"Summarize the following text:\n{transcript}"
        return self.chat_model.predict(prompt)

    def store_summary(self, video_url, summary):
        chunks = self.text_splitter.split_text(summary)
        self.db.add_texts(chunks, metadatas=[{"video_url": video_url}] * len(chunks))

    def query(self, question):
        retriever = self.db.as_retriever()
        chain = ConversationalRetrievalChain.from_llm(self.chat_model, retriever)
        response = chain.run(question)
        return response
