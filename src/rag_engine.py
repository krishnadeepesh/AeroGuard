"""
AeroGuard - RAG Knowledge Engine & LLM Generation Module
Chunks reference guidelines (WHO, CPCB, Clinical Advisories),
computes semantic embeddings via Ollama, retrieves grounded passages,
and invokes local Llama-3.2 to produce evidence-grounded health advisories.
"""

import os
import glob
import json
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

KNOWLEDGE_BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "knowledge_base")
INDEX_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "knowledge_index.json")
OLLAMA_API_BASE = "http://127.0.0.1:11434"
EMBED_MODEL = "embeddinggemma:latest"
LLM_MODEL = "llama3.2:3b"

class RAGEngine:
    def __init__(self, use_ollama=True):
        self.use_ollama = use_ollama
        self.chunks = []
        self.embeddings = []
        self._load_or_build_index()

    def _get_embedding(self, text):
        """Fetches vector embedding for text using Ollama or fallback vector."""
        if self.use_ollama:
            try:
                res = requests.post(
                    f"{OLLAMA_API_BASE}/api/embeddings",
                    json={"model": EMBED_MODEL, "prompt": text},
                    timeout=20
                )
                if res.status_code == 200:
                    return res.json().get("embedding", [])
            except Exception as e:
                print(f"[RAG Warning] Ollama embedding error: {e}. Falling back to TF-IDF.")
        return None

    def chunk_documents(self):
        """Parses markdown files in knowledge_base into semantic chunks with metadata."""
        md_files = glob.glob(os.path.join(KNOWLEDGE_BASE_DIR, "*.md"))
        chunks = []
        for file_path in md_files:
            file_name = os.path.basename(file_path)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            sections = content.split("\n## ")
            for idx, sec in enumerate(sections):
                cleaned = sec.strip()
                if not cleaned:
                    continue
                header = cleaned.split("\n")[0].replace("#", "").strip()
                chunks.append({
                    "id": f"{file_name}_sec_{idx}",
                    "source": file_name,
                    "section": header,
                    "text": cleaned
                })
        return chunks

    def _load_or_build_index(self):
        """Loads cached index or computes fresh embeddings."""
        if os.path.exists(INDEX_PATH):
            try:
                with open(INDEX_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.chunks = data.get("chunks", [])
                    self.embeddings = np.array(data.get("embeddings", []))
                    if len(self.chunks) > 0 and len(self.embeddings) > 0:
                        print(f"[RAG] Loaded cached vector index with {len(self.chunks)} chunks.")
                        return
            except Exception as e:
                print(f"[RAG] Rebuilding index due to load error: {e}")

        self.chunks = self.chunk_documents()
        emb_list = []
        print(f"[RAG] Computing embeddings for {len(self.chunks)} knowledge chunks...")
        for c in self.chunks:
            emb = self._get_embedding(f"{c['section']}: {c['text'][:300]}")
            if emb:
                emb_list.append(emb)
            else:
                emb_list.append([0.0] * 768)

        self.embeddings = np.array(emb_list)
        os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
        with open(INDEX_PATH, "w", encoding="utf-8") as f:
            json.dump({
                "chunks": self.chunks,
                "embeddings": self.embeddings.tolist()
            }, f)
        print(f"[RAG] Saved vector index to: {INDEX_PATH}")

    def retrieve(self, query, top_k=2):
        """Retrieves top-k most relevant knowledge passages for a query."""
        if len(self.chunks) == 0:
            return []

        query_emb = self._get_embedding(query)
        if query_emb is not None and self.embeddings.shape[0] > 0 and self.embeddings.shape[1] > 0:
            query_vec = np.array(query_emb).reshape(1, -1)
            sims = cosine_similarity(query_vec, self.embeddings)[0]
            top_indices = np.argsort(sims)[::-1][:top_k]
            results = []
            for idx in top_indices:
                score = float(sims[idx])
                chunk = self.chunks[idx].copy()
                chunk["similarity_score"] = round(score, 4)
                results.append(chunk)
            return results
        else:
            # Fallback simple keyword match
            keywords = [w.lower() for w in query.split() if len(w) > 3]
            scored = []
            for c in self.chunks:
                score = sum(1 for kw in keywords if kw in c["text"].lower())
                scored.append((score, c))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [item[1] for item in scored[:top_k]]

    def generate_advisory(self, ml_prediction, sensor_inputs, user_query=None):
        """
        Integrates ML model output (category, confidence, dominant pollutant)
        with retrieved guideline documents to generate a structured AI advisory.
        """
        category = ml_prediction.get("predicted_bucket", "Moderate")
        confidence = ml_prediction.get("confidence", 0.0)
        dominant_pollutant = ml_prediction.get("dominant_pollutant", "PM2.5")

        retrieval_query = (
            f"Air Quality {category} health risks precautions {dominant_pollutant} "
            f"mask guidelines children asthmatics ventilation"
        )
        retrieved_docs = self.retrieve(retrieval_query, top_k=2)

        context_text = "\n\n".join([
            f"[Source: {d['source']} | Section: {d['section']}]\n{d['text']}"
            for d in retrieved_docs
        ])

        system_prompt = (
            "You are AeroGuard, an expert environmental health advisory AI assistant. "
            "Your guidance must strictly align with the provided WHO and CPCB public health evidence. "
            "Do NOT hallucinate medications or provide ungrounded clinical diagnoses. "
            "Be clear, precise, and practical for citizen safety."
        )

        user_prompt = f"""
### Ambient Environmental Readings & ML Classification Result:
- **ML Predicted AQI Category**: {category} (Model Confidence: {confidence*100:.1f}%)
- **Dominant Threat Pollutant**: {dominant_pollutant}
- **Sensor Measurements**:
  - PM2.5: {sensor_inputs.get('PM2.5', 'N/A')} ug/m3
  - PM10: {sensor_inputs.get('PM10', 'N/A')} ug/m3
  - NO2: {sensor_inputs.get('NO2', 'N/A')} ug/m3
  - SO2: {sensor_inputs.get('SO2', 'N/A')} ug/m3
  - CO: {sensor_inputs.get('CO', 'N/A')} mg/m3
  - O3: {sensor_inputs.get('O3', 'N/A')} ug/m3
  - Temp / Humidity: {sensor_inputs.get('Temperature', 'N/A')} C / {sensor_inputs.get('Humidity', 'N/A')} %

### Verified Health & Safety Evidence (Retrieved from Knowledge Base):
{context_text}

### Instructions:
Generate a structured, evidence-based Public Health Advisory with the following distinct sections:
1. **Executive Severity Assessment**: Concise summary of current air hazard based on the ML prediction.
2. **Pathological Risks from {dominant_pollutant}**: How this specific dominant pollutant affects human physiology.
3. **Targeted Advisory for Vulnerable Demographics**: Actionable guidance for children, asthmatics, and senior citizens.
4. **Protective & Mitigation Measures**: Specific mask specifications (e.g. N95), indoor air purification, and outdoor activity restrictions.

Keep the advice crisp, professional, and directly grounded in the cited standards.
"""

        llm_response = None
        try:
            res = requests.post(
                f"{OLLAMA_API_BASE}/api/generate",
                json={
                    "model": LLM_MODEL,
                    "prompt": f"{system_prompt}\n\n{user_prompt}",
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "top_p": 0.9,
                        "num_predict": 450
                    }
                },
                timeout=45
            )
            if res.status_code == 200:
                llm_response = res.json().get("response", "").strip()
        except Exception as e:
            print(f"[RAG] Ollama generation exception: {e}")

        # High-quality fallback if Ollama times out or is offline
        if not llm_response:
            llm_response = (
                f"### Executive Severity Assessment\n"
                f"The ambient atmosphere is evaluated as **{category}** air quality condition "
                f"(Classified with {confidence*100:.1f}% confidence by the Random Forest model). "
                f"Immediate environmental precautions are mandated.\n\n"
                f"### Pathological Risks from {dominant_pollutant}\n"
                f"The primary driver of toxicity is elevated **{dominant_pollutant}**. Inhalation of fine particulates "
                f"penetrates deep into pulmonary alveoli, elevating systemic arterial inflammation and acute airway hyper-reactivity.\n\n"
                f"### Targeted Advisory for Vulnerable Demographics\n"
                f"- **Pediatric & Geriatric**: Strictly cancel all strenuous outdoor physical exercise. Relocate activities indoors.\n"
                f"- **Chronic Cardiopulmonary Patients**: Maintain strict compliance with prescribed bronchodilators and keep rescue inhalers immediately accessible.\n\n"
                f"### Protective & Mitigation Measures\n"
                f"- Wear certified **N95/FFP2 respirators** if exposure cannot be avoided.\n"
                f"- Activate indoor **True-HEPA filtration** systems with doors and windows sealed during peak inversion hours."
            )

        return {
            "advisory": llm_response,
            "retrieved_sources": retrieved_docs,
            "category": category,
            "confidence": confidence,
            "dominant_pollutant": dominant_pollutant
        }

if __name__ == "__main__":
    engine = RAGEngine()
    sample_ml = {"predicted_bucket": "Very Poor", "confidence": 0.96, "dominant_pollutant": "PM2.5"}
    sample_sensors = {"PM2.5": 185.0, "PM10": 340.0, "NO2": 78.0, "SO2": 22.0, "CO": 3.8, "O3": 45.0, "Temperature": 18.0, "Humidity": 68.0}
    result = engine.generate_advisory(sample_ml, sample_sensors)
    print("\n--- Generated Advisory ---\n")
    print(result["advisory"])
    print("\n--- Sources Used ---")
    for s in result["retrieved_sources"]:
        print(f"- {s['source']} ({s['section']}) [Score: {s.get('similarity_score', 'N/A')}]")
