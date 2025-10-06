# AUREA - Autonomous Ethical Reasoning Agent

AUREA è un sistema di intelligenza artificiale che integra modelli LLM e un grafo di conoscenza Neo4j per generare decisioni etiche basate su scenari. Il progetto è pensato per essere eseguito su robot NAO tramite ROS.

## Funzionalità principali

- Connessione e gestione del robot NAO tramite ROS
- Trascrizione audio in testo usando Whisper
- Generazione di query su Neo4j per recuperare informazioni pertinenti
- Spiegazioni basate su LLM multipli (GPT-4, Gemini, LLaMa, ApiFreeLLM)
- Modellazione socratica: genera domande e risposte per migliorare l’interpretabilità
- Giuria automatica per valutare le spiegazioni e selezionare il miglior modello

## Installazione

1. Clona la repository:
```bash
git clone https://github.com/DavideWho/AUREA.git
cd AUREA
pip install -r requirements.txt
OPENAI_API_KEY="la_tua_chiave_openai"
GEMINI_API_KEY="la_tua_chiave_gemini"
AWANLLM_API_KEY="la_tua_chiave_awanllm"
GROQ_API_KEY="la_tua_chiave_groq"
python aurea_script.py
jupyter notebook
# Apri il notebook corrispondente
