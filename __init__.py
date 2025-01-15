import sqlite3
import os
from pathlib import Path

def init_database():
    # Chemin de la base de données
    db_path = "data/vectors.db"
    
    # Créer le dossier data s'il n'existe pas
    Path("data").mkdir(exist_ok=True)
    
    # Supprimer l'ancienne base de données si elle existe
    if os.path.exists(db_path):
        os.remove(db_path)
    
    # Créer et initialiser la nouvelle base de données
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript("""
            -- Table principale des documents
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY,
                content TEXT NOT NULL,
                vector BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata JSON
            );
            
            -- Index sur la date de création
            CREATE INDEX IF NOT EXISTS idx_documents_created 
            ON documents(created_at);
            
            -- Table des métadonnées
            CREATE TABLE IF NOT EXISTS document_metadata (
                document_id INTEGER,
                key TEXT,
                value TEXT,
                FOREIGN KEY(document_id) REFERENCES documents(id),
                PRIMARY KEY(document_id, key)
            );
        """)
        conn.commit()
        print("Base de données initialisée avec succès!")
        
    except Exception as e:
        print(f"Erreur lors de l'initialisation de la base de données: {str(e)}")
        raise
        
    finally:
        conn.close()

if __name__ == "__main__":
    init_database()