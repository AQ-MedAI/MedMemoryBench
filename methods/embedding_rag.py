"""Embedding RAG Agent - vector retrieval with FAISS."""

import logging
from typing import Optional, List

from .base import BaseAgent, MemoryBuildResult, AgentResponse
from utils.llm_client import create_llm_client, format_messages, BaseLLMClient

logger = logging.getLogger(__name__)


class BaseLocalEmbeddings:
    """Base class for local embedding models with mean pooling."""

    def __init__(self, model_name: str):
        import torch
        from transformers import AutoTokenizer, AutoModel
        import torch.nn.functional as F

        self.torch = torch
        self.F = F

        logger.info(f"[Embedding] Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, device_map="auto")
        self.model.eval()

    def _get_embedding(self, outputs, attention_mask):
        """Override in subclass for different pooling strategies."""
        raise NotImplementedError

    def _embed_single(self, text: str) -> List[float]:
        """Embed a single text."""
        inputs = self.tokenizer(
            text, padding=True, truncation=True, return_tensors='pt'
        ).to(self.model.device)

        with self.torch.no_grad():
            outputs = self.model(**inputs)

        embedding = self._get_embedding(outputs, inputs['attention_mask'])
        embedding = self.F.normalize(embedding, p=2, dim=1)
        return embedding.cpu().numpy()[0].tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Batch embed documents."""
        from tqdm import tqdm
        return [self._embed_single(text) for text in tqdm(texts, desc="Embedding")]

    def embed_query(self, text: str) -> List[float]:
        """Embed single query."""
        return self._embed_single(text)


class ContrieverEmbeddings(BaseLocalEmbeddings):
    """Facebook Contriever model using CLS token pooling."""

    def __init__(self, model_name: str = "facebook/contriever"):
        assert "contriever" in model_name.lower(), "Model name must contain 'contriever'"
        super().__init__(model_name)

    def _get_embedding(self, outputs, attention_mask):
        return outputs.last_hidden_state[:, 0, :]


class MeanPoolingEmbeddings(BaseLocalEmbeddings):
    """Base class for models using mean pooling."""

    def _get_embedding(self, outputs, attention_mask):
        last_hidden = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        sum_embeddings = (last_hidden * mask_expanded).sum(dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        return sum_embeddings / sum_mask


class Qwen3EmbeddingEmbeddings(MeanPoolingEmbeddings):
    """Qwen3-Embedding model wrapper."""

    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-4B"):
        super().__init__(model_name)


class NVEmbedEmbeddings(MeanPoolingEmbeddings):
    """NVIDIA NV-Embed model wrapper."""

    def __init__(self, model_name: str = "nvidia/NV-Embed-v2"):
        super().__init__(model_name)


class EmbeddingRAGAgent(BaseAgent):
    """Embedding RAG Agent using vector embeddings and FAISS for retrieval."""

    METHOD_TYPE = "rag"
    DEFAULT_MAX_CONTEXT_TOKENS = 120000

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 1.0,
        max_tokens: int = 2000,
        provider: str = "openai",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        embedding_model: str = "text-embedding-3-small",
        embedding_provider: str = "openai",
        embedding_model_path: Optional[str] = None,
        top_k: int = 5,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        max_context_tokens: Optional[int] = None,
        **kwargs
    ):
        super().__init__(model, temperature, max_tokens, **kwargs)

        self.embedding_model = embedding_model
        self.embedding_provider = embedding_provider
        self.embedding_model_path = embedding_model_path
        self.top_k = top_k
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_context_tokens = max_context_tokens or self.DEFAULT_MAX_CONTEXT_TOKENS

        self._api_key = api_key
        self._base_url = base_url

        self._llm_client: BaseLLMClient = create_llm_client(
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            base_url=base_url,
        )

        self._vectorstore = None
        self._embedding_model_instance = None
        self._chunks: List[str] = []

    def _split_text_into_chunks(self, text: str) -> List[str]:
        """Split text into token-based chunks."""
        if not text.strip():
            return []

        tokens = self._tokenizer.encode(text)
        if len(tokens) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunks.append(self._tokenizer.decode(tokens[start:end]))
            start = end - self.chunk_overlap
            if end == len(tokens):
                break

        return chunks

    def _get_embedding_model(self):
        """Get or create embedding model instance."""
        if self._embedding_model_instance is not None:
            return self._embedding_model_instance

        model_name_lower = self.embedding_model.lower()
        provider_lower = self.embedding_provider.lower()
        model_path = self.embedding_model_path or self.embedding_model

        if provider_lower == "contriever" or "contriever" in model_name_lower:
            self._embedding_model_instance = ContrieverEmbeddings(model_path)
        elif provider_lower in ("qwen3", "qwen3-embedding") or "qwen3-embedding" in model_name_lower:
            self._embedding_model_instance = Qwen3EmbeddingEmbeddings(model_path)
        elif provider_lower in ("nv-embed", "nvidia") or "nv-embed" in model_name_lower:
            self._embedding_model_instance = NVEmbedEmbeddings(model_path)
        elif provider_lower == "openai":
            from langchain_openai import OpenAIEmbeddings
            kwargs = {"model": self.embedding_model}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._embedding_model_instance = OpenAIEmbeddings(**kwargs)
        elif provider_lower in ("local", "huggingface"):
            from langchain_huggingface import HuggingFaceEmbeddings
            self._embedding_model_instance = HuggingFaceEmbeddings(
                model_name=model_path,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        else:
            raise ValueError(f"Unsupported embedding provider: {self.embedding_provider}")

        return self._embedding_model_instance

    def _build_vectorstore(self) -> None:
        """Build FAISS vector store from chunks."""
        if not self._chunks:
            return

        from langchain_community.vectorstores import FAISS
        from langchain_core.documents import Document

        documents = [
            Document(page_content=chunk, metadata={"index": i})
            for i, chunk in enumerate(self._chunks)
        ]

        self._vectorstore = FAISS.from_documents(documents, self._get_embedding_model())

    def memorize(self, text: str, **kwargs) -> MemoryBuildResult:
        """Add text to memory and build vector store."""
        self._memory_chunks.append(text)
        new_chunks = self._split_text_into_chunks(text)
        self._chunks.extend(new_chunks)
        self._is_initialized = True
        self._build_vectorstore()

        passages = [
            {"index": len(self._chunks) - len(new_chunks) + i, "content": chunk[:200], "tokens_est": len(chunk) // 4}
            for i, chunk in enumerate(new_chunks)
        ]

        return MemoryBuildResult(
            success=True,
            method="embedding_rag",
            action="build_vectorstore",
            input_content=text,
            stored_content=text,
            all_passages=passages,
            memory_entries=[{"type": "vector_chunk", "count": len(new_chunks)}],
            chunk_count=len(self._chunks),
            extra={"original_doc_count": len(self._memory_chunks), "new_chunks": len(new_chunks)},
        )

    def _retrieve(self, query: str) -> List[str]:
        """Retrieve relevant documents using similarity search."""
        if self._vectorstore is None:
            return []
        results = self._vectorstore.similarity_search(query, k=self.top_k)
        return [doc.page_content for doc in results]

    def query(
        self,
        question: str,
        system_message: Optional[str] = None,
        **kwargs
    ) -> AgentResponse:
        """Query the agent with embedding retrieval."""
        retrieved_docs = self._retrieve(question)
        truncated_docs = self.truncate_docs_to_context(
            docs=retrieved_docs,
            question=question,
            system_message=system_message,
            max_context_tokens=self.max_context_tokens,
        )

        if truncated_docs:
            context = "\n\n".join([
                f"[Memory {i+1}]\n{doc}"
                for i, doc in enumerate(truncated_docs)
            ])
            full_message = f"{context}\n\n{question}"
        else:
            full_message = question

        messages = format_messages(full_message, system_message)
        response = self._llm_client.chat(messages)

        formatted_memories = [
            {"memory": doc[:500] + "..." if len(doc) > 500 else doc, "type": "embedding_retrieval"}
            for doc in truncated_docs
        ]

        return AgentResponse(
            output=response.content,
            query_time=0.0,
            retrieved_count=len(truncated_docs),
            retrieved_memories=formatted_memories,
            extra={
                "method": "embedding_rag",
                "embedding_model": self.embedding_model,
                "embedding_provider": self.embedding_provider,
                "original_retrieved_count": len(retrieved_docs),
                "truncated_count": len(truncated_docs),
            }
        )

    def reset(self) -> None:
        """Reset agent state."""
        super().reset()
        self._vectorstore = None
        self._chunks = []

    @property
    def has_vectorstore(self) -> bool:
        """Check if vector store is built."""
        return self._vectorstore is not None

    @property
    def chunk_count(self) -> int:
        """Get current chunk count."""
        return len(self._chunks)
