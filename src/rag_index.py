import os
import json
import numpy as np

class EHTCatalogRAGIndex:
    """
    Implements a simple vector index database (RAG) for the EHT Black Hole observations.
    Ref: DAG 04 presentation concept.
    """
    def __init__(self, index_dir=None):
        if index_dir is None:
            self.index_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        else:
            self.index_dir = index_dir
            
        self.index_path = os.path.join(self.index_dir, 'eht_rag_index.json')
        self.vector_dim = 16 # Small embedding size for didactic speed
        self._init_index()
        
    def _init_index(self):
        if not os.path.exists(self.index_path):
            initial_db = {
                'documents': [],
                'embeddings': []
            }
            with open(self.index_path, 'w') as f:
                json.dump(initial_db, f, indent=4)
                
    def _read_index(self):
        with open(self.index_path, 'r') as f:
            return json.load(f)
            
    def _write_index(self, db):
        with open(self.index_path, 'w') as f:
            json.dump(db, f, indent=4)
            
    def generate_fake_embedding(self, text):
        """
        Generates a deterministic 16-dimensional embedding vector.
        Uses a weighted ASCII sum and sine scaling for cross-platform matching.
        """
        import math
        vector = []
        for i in range(self.vector_dim):
            val = sum(ord(c) * (i + 1) * (idx + 1) for idx, c in enumerate(text))
            vector.append(math.sin(val) * 0.5)
            
        arr = np.array(vector)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr.tolist()

    def add_model_to_index(self, run_id, parameters, metrics, fit_result):
        """
        Chunks the model metadata, generates embeddings, and adds them to the RAG index database.
        """
        db = self._read_index()
        
        # 1. Create document chunks
        chunks = [
            {
                'id': f"{run_id}_chunk_summary",
                'text': f"O buraco negro supermassivo M87* foi observado e imageado na run {run_id}. A massa estimada é de {fit_result['estimated_mass_10_9']} bilhões de massas solares, com spin adimensional a = {fit_result['estimated_spin']:.2f}.",
                'metadata': {'run_id': run_id, 'type': 'summary'}
            },
            {
                'id': f"{run_id}_chunk_quality",
                'text': f"A imagem reconstruída de M87* na run {run_id} atingiu um escore de fidelidade NCC de {metrics['fidelity_score']*100:.2f}% e um Erro Quadrático Médio (MSE) de {metrics['mse']:.6e} contra a Relatividade Geral.",
                'metadata': {'run_id': run_id, 'type': 'quality'}
            },
            {
                'id': f"{run_id}_chunk_cal",
                'text': f"O processamento na run {run_id} utilizou calibração baseada em Fase de Fechamento (Closure Phase) de 1.3mm, com parâmetros de regularização adicionais: alpha_tv = {parameters['alpha_tv']} e alpha_entropy = {parameters['alpha_entropy']}.",
                'metadata': {'run_id': run_id, 'type': 'calibration'}
            }
        ]
        
        # Avoid duplicate documents
        existing_ids = {doc['id'] for doc in db['documents']}
        
        for chunk in chunks:
            if chunk['id'] not in existing_ids:
                emb = self.generate_fake_embedding(chunk['text'])
                db['documents'].append(chunk)
                db['embeddings'].append({
                    'id': chunk['id'],
                    'vector': emb
                })
                
        self._write_index(db)
        print(f"Added {len(chunks)} document chunks to RAG Index for run {run_id}.")
        return len(chunks)

    def query_index(self, query_text, top_k=2):
        """
        Queries the vector index for similar documents using cosine similarity.
        """
        db = self._read_index()
        if not db['documents']:
            return []
            
        q_emb = np.array(self.generate_fake_embedding(query_text))
        
        results = []
        for doc, emb_data in zip(db['documents'], db['embeddings']):
            doc_emb = np.array(emb_data['vector'])
            
            # Cosine similarity: (A . B) / (|A| |B|)
            # Since vectors are normalized, it is just the dot product
            similarity = float(np.dot(q_emb, doc_emb))
            
            results.append({
                'document': doc,
                'similarity': similarity
            })
            
        # Sort by similarity descending
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]

if __name__ == "__main__":
    idx = EHTCatalogRAGIndex()
    idx.add_model_to_index(
        run_id='run_test',
        parameters={'alpha_tv': 0.05, 'alpha_entropy': 0.005},
        metrics={'fidelity_score': 0.965, 'mse': 8.5e-8},
        fit_result={'estimated_mass_10_9': 6.5, 'estimated_spin': 0.5}
    )
    
    # Query test
    res = idx.query_index("Qual é a massa de M87*?")
    for r in res:
        print(f"Sim: {r['similarity']:.4f} | Text: {r['document']['text']}")
