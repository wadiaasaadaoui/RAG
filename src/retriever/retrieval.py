import sqlite3
from contextlib import contextmanager
from typing import List, Tuple, Dict, Any, Generator
import numpy as np
from scipy.spatial.distance import cosine
import json
import os
import yaml
from datetime import datetime
from dataclasses import dataclass
from threading import Lock

@dataclass
class VectorStoreConfig:
    db_path: str
    chunk_size: int
    similarity_threshold: float
    max_results: int
    connection_timeout: int

class ConnectionPool:
    def __init__(self, db_path: str, max_connections: int = 5, timeout: int = 30):
        self.db_path = db_path
        self.max_connections = max_connections
        self.timeout = timeout
        self.connections: List[sqlite3.Connection] = []
        self.available: List[bool] = []
        self.lock = Lock()
        
    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        connection = self._acquire_connection()
        try:
            yield connection
        finally:
            self._release_connection(connection)
            
    def _acquire_connection(self) -> sqlite3.Connection:
        with self.lock:
            # Chercher une connexion disponible
            for i, available in enumerate(self.available):
                if available:
                    self.available[i] = False
                    return self.connections[i]
            
            # Créer une nouvelle connexion si possible
            if len(self.connections) < self.max_connections:
                connection = sqlite3.connect(
                    self.db_path,
                    timeout=self.timeout,
                    detect_types=sqlite3.PARSE_DECLTYPES
                )
                connection.row_factory = sqlite3.Row
                self.connections.append(connection)
                self.available.append(False)
                return connection
                
            raise RuntimeError("No available connections")
            
    def _release_connection(self, connection: sqlite3.Connection):
        with self.lock:
            idx = self.connections.index(connection)
            self.available[idx] = True

class Retriever:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)
        self.connection_pool = ConnectionPool(
            self.config.db_path,
            max_connections=5,
            timeout=self.config.connection_timeout
        )
        self.init_db()
        
    def _load_config(self, config_path: str) -> VectorStoreConfig:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            vector_store = config['vector_store']
            return VectorStoreConfig(
                db_path=vector_store['db_path'],
                chunk_size=vector_store['settings']['chunk_size'],
                similarity_threshold=vector_store['settings']['similarity_threshold'],
                max_results=vector_store['settings']['max_results'],
                connection_timeout=vector_store['settings']['connection_timeout']
            )
    
    def init_db(self):
        with self.connection_pool.get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY,
                    content TEXT NOT NULL,
                    vector BLOB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata JSON
                );
                
                CREATE INDEX IF NOT EXISTS idx_documents_created 
                ON documents(created_at);
                
                CREATE TABLE IF NOT EXISTS document_metadata (
                    document_id INTEGER,
                    key TEXT,
                    value TEXT,
                    FOREIGN KEY(document_id) REFERENCES documents(id),
                    PRIMARY KEY(document_id, key)
                );
            """)
            conn.commit()
    
    def add_documents(self, vectors: np.ndarray, documents: List[str], metadata: List[Dict] = None):
        if metadata is None:
            metadata = [{}] * len(documents)
            
        with self.connection_pool.get_connection() as conn:
            for doc, vec, meta in zip(documents, vectors, metadata):
                # Convertir le vecteur en bytes pour un stockage efficace
                vec_bytes = vec.tobytes()
                
                cursor = conn.execute(
                    "INSERT INTO documents (content, vector, metadata) VALUES (?, ?, ?)",
                    (doc, vec_bytes, json.dumps(meta))
                )
                doc_id = cursor.lastrowid
                
                # Stocker les métadonnées de manière structurée
                for key, value in meta.items():
                    conn.execute(
                        "INSERT INTO document_metadata (document_id, key, value) VALUES (?, ?, ?)",
                        (doc_id, key, str(value))
                    )
            conn.commit()
    
    def retrieve(
        self,
        query_vector: np.ndarray,
        k: int = None,
        threshold: float = None,
        filter_metadata: Dict[str, str] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        k = k or self.config.max_results
        threshold = threshold or self.config.similarity_threshold
        
        with self.connection_pool.get_connection() as conn:
            # Construire la requête de base
            query = """
                SELECT d.id, d.content, d.vector, d.metadata, d.created_at
                FROM documents d
            """
            
            params = []
            if filter_metadata:
                # Ajouter les conditions de filtrage sur les métadonnées
                metadata_conditions = []
                for key, value in filter_metadata.items():
                    query += """
                        JOIN document_metadata dm_{key}
                        ON d.id = dm_{key}.document_id
                        AND dm_{key}.key = ?
                        AND dm_{key}.value = ?
                    """.format(key=key)
                    params.extend([key, value])
            
            cursor = conn.execute(query, params)
            results = []
            
            for row in cursor:
                vec = np.frombuffer(row['vector'], dtype=np.float32)
                similarity = 1 - cosine(query_vector, vec)
                
                if similarity >= threshold:
                    results.append((
                        row['content'],
                        similarity,
                        {
                            'id': row['id'],
                            'created_at': row['created_at'],
                            'metadata': json.loads(row['metadata'])
                        }
                    ))
            
            # Trier par similarité et retourner les k meilleurs résultats
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:k]
    
    def __del__(self):
        # Fermer toutes les connexions lors de la destruction de l'objet
        if hasattr(self, 'connection_pool'):
            for conn in self.connection_pool.connections:
                conn.close()