from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import os, uuid

class VectorStore(ABC):
    """Abstract base class for vector stores."""
    
    @abstractmethod
    def add_prompts(self,  record_ids: List[str], task_descriptions: List[str],\
                    prompts: List[str]) -> None:
        """
        An abstract method to add prompts, task descriptions, and associated record IDs.

        Summary:
        This method serves as a contract for subclasses to implement functionality that
        associates a list of prompts with their corresponding task descriptions and record
        IDs. It does not provide an implementation by itself and must be implemented by any
        concrete subclass.

        Args:
            record_ids: A list of strings representing unique identifiers for records.
            task_descriptions: A list of strings containing task descriptions corresponding
                to each record.
            prompts: A list of strings holding the prompt details to be associated with the
                records.

        Returns:
            None
        """
        pass
    
    @abstractmethod
    def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents.
        
        Args:
            query: Query string to search for
            n_results: Number of results to return
            
        Returns:
            List of similar documents
        """
        pass

class PromptChromaStore(VectorStore):
    """ChromaDB implementation of VectorStore."""
    
    def __init__(self,persist_directory: str = "./prompts", collection_name :str ="prompt",\
                 embedding=None):
        """Initialize ChromaDB vector store.
        
        Args:
            persist_directory: Directory to persist the database
        """
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,allow_reset=True

            )
        )
        
        # Use default embedding function (all-MiniLM-L6-v2)
        self.embedding_function = embedding_functions.DefaultEmbeddingFunction() if not embedding else embedding
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )

    def add_prompts(self,  task_descriptions: List[str],\
                    prompts: List[str], record_ids: List[str] =None) -> None:

        """
        Add prompts to the vector store.

        Args:
            record_ids: List of unique identifiers for each prompt record.
            task_descriptions: List of descriptions of the tasks associated with each prompt.
            prompts: List of actual prompt text content to be stored.

        Raises:
            ValueError: If the lengths of record_ids, task_descriptions, and prompts don't match.
        """
        # Validate input

        if record_ids is None:
            record_ids = [str(uuid.uuid4()) for _ in range(len(task_descriptions))]

        if not (len(record_ids) == len(task_descriptions) == len(prompts)):
            raise ValueError("The lengths of record_ids, task_descriptions, and prompts must be equal")


        # Prepare data for the vector store
        ids: List[str] = []
        texts: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for i in range(len(record_ids)):
            ids.append(record_ids[i])
            texts.append(task_descriptions[i])  # Using task descriptions as the text to embed

            # Create metadata dictionary
            metadata: Dict[str, Any] = {
                "prompt": prompts[i],
            }

            metadatas.append(metadata)

        # Add the documents to the vector store
        try:
            # Assuming self.collection is a ChromaDB collection
            self.collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas
            )
            print(f"Successfully added {len(ids)} prompts to the vector store.")
        except Exception as e:
            print(f"Error adding prompts to vector store: {str(e)}")
            raise


    def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for similar tasks and return their associated prompts.
        
        Args:
            query: Task description to search for
            n_results: Number of results to return
            
        Returns:
            List of dictionaries containing:
                - task: The original task
                - prompt: The prompt used
                - model: The model used
                - metadata: Additional metadata
                - distance: Similarity score
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        similar_documents = []
        for i in range(len(results["ids"][0])):
            doc = {
                "task": results["documents"][0][i],
                "prompt": results["metadatas"][0][i]["prompt"],
                "distance": results["distances"][0][i],
                "record_id": results["ids"][0][i],
            }
            # Add any additional metadata
            for key, value in results["metadatas"][0][i].items():
                if key not in ["prompt", "model"]:
                    doc[key] = value
            similar_documents.append(doc)
        
        return similar_documents

    def close(self):
        # we just delete the client
        del self.collection
        del self.embedding_function
        del self.client

    def clear(self):
        # delete all records
        self.client.reset()



if __name__ =="__main__":
    pass



