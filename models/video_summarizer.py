from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Milvus
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import SystemMessage, HumanMessage, AIMessage

class VideoSummarizer:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.db = Milvus(
            embedding_function=self.embeddings,
            connection_args={"host": "localhost", "port": 19530},
            collection_name="video_summaries",
        )
        # self.llm = OpenAI(temperature=0, model_name="gpt-4")
        self.chat_model = ChatOpenAI(temperature=0, model_name="gpt-4")
        self.text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    # def fetch_transcript(self, video_url):
    #     video_id = video_url.split("v=")[-1]
    #     print(video_id)
    #     transcript = YouTubeTranscriptApi.get_transcript(video_id)
    #     return " ".join([segment['text'] for segment in transcript])

    # def fetch_transcript(self, video_url):
    #     try:
    #         video_id = video_url.split("v=")[-1].split("&")[0]  # Extract video ID
    #         transcript = YouTubeTranscriptApi.get_transcript(video_id)
    #         return " ".join([segment['text'] for segment in transcript])
    #     except Exception as e:
    #         raise ValueError(f"Error fetching transcript: {str(e)}")

    def fetch_transcript(self, video_url):
        try:
            if "v=" in video_url:
                video_id = video_url.split("v=")[1].split("&")[0]
            else:
                video_id = video_url

            # Attempt to fetch the transcript
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            return " ".join([segment['text'] for segment in transcript])
        except Exception as e:
            print(f"Error fetching transcript for video {video_url}: {e}")
            raise ValueError(f"Error fetching transcript: {str(e)}")

    # def summarize_transcript(self, transcript):
    #     prompt = f"Summarize the following text:\\n{transcript}"
    #     return self.chat_model(prompt)

    def summarize_transcript(self, transcript):
        # Prepare the input as a list of messages
        messages = [
            SystemMessage(content="You are a helpful assistant who summarizes text."),
            HumanMessage(content=f"Summarize the following text:\n{transcript}")
        ]
        # Use the chat model to generate a response
        return self.chat_model(messages).content
    def store_summary(self, video_url, summary):
        chunks = self.text_splitter.split_text(summary)
        self.db.add_texts(chunks, metadatas=[{"video_url": video_url}] * len(chunks))

    # def query(self, question):
    #     retriever = self.db.as_retriever()
    #     chain = ConversationalRetrievalChain.from_llm(self.chat_model, retriever)
    #     response = chain.run(question)
    #     return response


    def query(self, question, chat_history=[]):
        retriever = self.db.as_retriever()
        chain = ConversationalRetrievalChain.from_llm(self.chat_model, retriever)

        # Convert chat history to a list of ChatMessages
        formatted_chat_history = [
            HumanMessage(content=message["content"]) if message["role"] == "user" else AIMessage(
                content=message["content"])
            for message in chat_history
        ]

        # Pass the question and formatted chat history
        response = chain({"question": question, "chat_history": formatted_chat_history})
        return response["answer"]
