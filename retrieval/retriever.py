from sentence_transformers import SentenceTransformer
import numpy as np
import faiss


class Retriever:
    """
    Semantic Retriever for Text and Visual Documents.

    This class builds a FAISS index over text embeddings generated
    using a SentenceTransformer model. It supports retrieval across:

        - plain text documents
        - visual documents (figures with extracted data)

    The retriever enables unified search across textual and visual
    information extracted from scientific papers.

    Attributes
    ----------
    model : SentenceTransformer
        Embedding model used to encode documents and queries.

    docs : list[str]
        Text representation of indexed documents.

    images : list[str]
        Associated image paths for visual documents (None for text docs).

    data : list[dict]
        Structured metadata extracted from visual figures.

    index : faiss.IndexFlatL2
        FAISS vector index for nearest neighbor search.
    """

    def __init__(self, texts, visual_docs=None):
        """
        Initialize the semantic retriever and build the FAISS index.

        Parameters
        ----------
        texts : list[str]
            Plain text documents to index.

        visual_docs : list[dict], optional
            Visual documents containing extracted data.

            Format:
                {
                    "text": str,
                    "image": str,
                    "data": dict
                }
        """

        # Sentence embedding model
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # Storage containers
        self.docs = []
        self.images = []
        self.data = []

        # Add text-only documents
        for t in texts:
            self.docs.append(t)
            self.images.append(None)
            self.data.append(None)

        # Add visual documents
        if visual_docs:
            for v in visual_docs:
                self.docs.append(v["text"])
                self.images.append(v["image"])
                self.data.append(v["data"])

        # Generate embeddings for all documents
        embeddings = self.model.encode(self.docs)

        embeddings = np.array(embeddings)

        dim = embeddings.shape[1]

        # Create FAISS index
        self.index = faiss.IndexFlatL2(dim)

        # Add embeddings to index
        self.index.add(embeddings)

    def search(self, query, k=30):
        """
        Retrieve the top-k most relevant documents for a query.

        The query is embedded using the same sentence transformer,
        and FAISS performs nearest neighbor search.

        Parameters
        ----------
        query : str
            Natural language search query.

        k : int
            Number of top results to retrieve.

        Returns
        -------
        list[dict]
            Retrieved results in the format:

            {
                "text": str,
                "image": str | None,
                "data": dict | None
            }
        """

        # Encode query
        q = self.model.encode([query])

        # Perform nearest neighbor search
        D, I = self.index.search(q, k)

        results = []

        for idx in I[0]:

            results.append({
                "text": self.docs[idx],
                "image": self.images[idx],
                "data": self.data[idx]
            })

        return results