import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # AWS Bedrock
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    BEDROCK_MODEL_ID: str = os.getenv(
        "BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-20250514-v1:0"
    )
    BEDROCK_EMBEDDING_MODEL_ID: str = os.getenv(
        "BEDROCK_EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0"
    )

    # Pinecone
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "financial-policy-index")

    # LangSmith
    LANGCHAIN_TRACING_V2: str = os.getenv("LANGCHAIN_TRACING_V2", "true")
    LANGCHAIN_API_KEY: str = os.getenv("LANGCHAIN_API_KEY", "")
    LANGCHAIN_PROJECT: str = os.getenv("LANGCHAIN_PROJECT", "financial-policy-qa")

    # Retrieval
    BM25_PARAMS_PATH: str = os.getenv("BM25_PARAMS_PATH", "bm25_params.json")
    TOP_K_DENSE: int = 10
    TOP_K_SPARSE: int = 10
    TOP_K_FINAL: int = 5
    ALPHA: float = 0.7          # weight for dense vs sparse (1.0 = pure dense)
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64

    # App
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")


config = Config()
