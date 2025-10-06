# %%
from neo4j import GraphDatabase
from neo4j.graph import Node, Relationship, Path
import google.generativeai as genai
import json
import openai
from groq import Groq
import pandas as pd
import requests
import requests
import os
import io
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk import pos_tag, WordNetLemmatizer
stop_words = set(stopwords.words('english'))


# %%
# === Configurazione NEO4J ===

uri = "NEO4J_URI"
username = "NEO4J_USERNAME"
password = "NEO4J_PASSWORD"
driver = GraphDatabase.driver(uri, auth=(username, password))
# === Chiave per awllm ===
AWANLLM_API_KEY = "AWANLLM_API_KEY"
# === Configurazione GOOGLE GEMINI ===
gemini_api_key = "GEMINI_API_KEY"
genai.configure(api_key=gemini_api_key)
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

# == Configurazione GPT-4 ==
openai.api_key = "GPT4_API_KEY"

# == Configurazione LLaMa ==
groq_api_key = "GROQ_API_KEY"
llama = Groq(api_key=groq_api_key)

def estrai_struttura_grafo(driver):
    with driver.session() as session:
        # Estrai nodi e proprietà
        nodi = session.run("""
            CALL db.schema.nodeTypeProperties() 
            YIELD nodeType, propertyName, propertyTypes 
            RETURN toString(nodeType) AS label, 
                   collect(DISTINCT {name: propertyName, types: propertyTypes}) AS properties
        """)
        struttura_nodi = [
            {"label": record["label"], "properties": record["properties"]}
            for record in nodi
        ]

        # Estrai relazioni esistenti tramite MATCH
        rels = session.run("""
            MATCH (a)-[r]->(b)
            RETURN DISTINCT 
                type(r) AS type, 
                labels(a)[0] AS fromType, 
                labels(b)[0] AS toType
        """)
        struttura_rels = [
            {
                "type": record["type"],
                "from": record["fromType"],
                "to": record["toType"]
            }
            for record in rels
        ]

        return {"nodes": struttura_nodi, "relationships": struttura_rels}
def load_struttura(struttura_grafo): # === Load del file di struttura ===
    try:
        with open("struttura_grafo.json", "r", encoding="utf-8") as f:
            struttura_grafo = json.load(f)
    except UnicodeDecodeError:
        with open("struttura_grafo.json", "r", encoding="latin-1") as f:
            struttura_grafo = json.load(f)

    # === Prompt descrizione struttura grafo ===
    schema_testuale = ""     
    """
    Tipi di nodi e proprietà:
    """
    for nodo in struttura_grafo["nodes"]:
        label = nodo["label"]
        # Estrai solo i nomi delle proprietà se sono dizionari
        props = ", ".join(p["propertyName"] if isinstance(p, dict) and "propertyName" in p else str(p) for p in nodo["properties"])
        schema_testuale += f"- {label} (proprietà: {props})\n"

    #schema_testuale += "\nTipi di relazioni (con direzioni):\n"
    for rel in struttura_grafo["relationships"]:
        schema_testuale += f"- ({rel['from']})-[:{rel['type']}]->({rel['to']})\n"

    #print("Schema del grafo:\n", schema_testuale)
    return schema_testuale
# %%
def spiega_con_gpt(prompt, max_tokens=100):
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content.strip()
def gemini_wrapper(prompt, gemini_model=gemini_model):
    response = gemini_model.generate_content(
        prompt,
        generation_config={
            "temperature": 0.5,
            "max_output_tokens": 100,
            "top_p": 1,
            "top_k": 40
        }
    )
    return response.text.strip()
def llama_wrapper(prompt):
    completion = llama.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
        max_completion_tokens=100,
        top_p=1,
        stream=False
    )
    return completion.choices[0].message.content.strip()
def apifreellm_wrapper(prompt, temperature=0.5, max_tokens=100, top_p=1.0, top_k=40):
    """
    Wrapper per l'API di ApiFreeLLM.
    
    Args:
        prompt (str): Il testo del prompt da inviare al modello.
        temperature (float): Controlla la creatività del modello (non sempre supportato).
        max_tokens (int): Numero massimo di token di output desiderati.
        top_p (float): Nucleus sampling parameter (se supportato).
        top_k (int): Numero massimo di candidati considerati (se supportato).

    Returns:
        str: Risposta testuale dal modello oppure messaggio di errore.
    """
    url = "https://apifreellm.com/api/chat"
    headers = {"Content-Type": "application/json"}
    
    data = {
        "message": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "top_k": top_k
    }
    
    try:
        resp = requests.post(url, headers=headers, json=data, timeout=30)
        resp.raise_for_status()
        js = resp.json()
        
        if js.get('status') == 'success':
            return js['response'].strip()
        else:
            return f"Error: {js.get('error', 'Unknown error')} (status: {js.get('status')})"
    
    except requests.RequestException as e:
        return f"HTTP Request failed: {e}"
  
def awanllm_wrapper(prompt, model="Meta-Llama-3-8B-Instruct", max_tokens=1024, temperature=0.7, top_p=0.9, top_k=40, repetition_penalty=1.1):
    """
    Wrapper per l'API AwanLLM.
    
    Args:
        prompt (str): Il prompt da inviare al modello.
        model (str): Il modello da usare.
        max_tokens (int): Numero massimo di token di output.
        temperature (float): Creatività del modello.
        top_p (float): Nucleus sampling parameter.
        top_k (int): Numero massimo di candidati considerati.
        repetition_penalty (float): Penalità per ripetizioni.
        
    Returns:
        str: Risposta testuale dal modello.
    """
    url = "https://api.awanllm.com/v1/completions"
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f"Bearer {AWANLLM_API_KEY}"
    }
    
    payload = {
        "model": model,
        "prompt": f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are an assistant AI.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        "repetition_penalty": repetition_penalty,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_tokens": max_tokens,
        "stream": False  # per ricevere tutto in una volta
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        js = response.json()
        # AwanLLM restituisce tipicamente la risposta nel campo 'text' del primo item di 'choices'
        return js['choices'][0]['text'].strip()
    except requests.RequestException as e:
        return f"HTTP Request failed: {e}"
    except KeyError:
        return f"Unexpected response format: {js}"



# %% [markdown]
# **Snapshot dinamico**

# %%
def genera_query_KB(scenario, schema_testuale): 
    return f"""
        Given the structure of the Neo4j knowledge graph: {schema_testuale}

        And the following scenario: {scenario}
        
        Generate a Cypher query that:
        - Retrieves relevant nodes and relationships
        - Shows the properties of the nodes and the labels of the relationships
        - Searches for similarities in:
        - **keywords** or **concepts** mentioned in the scenario;
        - **types of actions** taken;
        - **results or impacts**;
        - **associated principles** or **moral values**;
        - **contexts** or types of people involved.
        - no introduction or explanation, just the Cypher query, no //comment.
        - BE CAREFUL: GRAPH CONTAIN ONLY ITALIAN WORDS
        """
def generate_cypher_query(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    words = [w.lower() for w in tokens if w.isalnum()]
    content_words = [w for w in words if w not in stop_words]

    bigrams = [' '.join(bg) for bg in ngrams(content_words, 2)]
    trigrams = [' '.join(tg) for tg in ngrams(content_words, 3)]
    all_terms = content_words + bigrams + trigrams

    words_cypher_list = ', '.join(f'"{term}"' for term in all_terms)

    cypher_query = f"""
UNWIND [{words_cypher_list}] AS word
MATCH (n)
WHERE ANY(propKey IN keys(n)
          WHERE n[propKey] IS NOT NULL
            AND toString(n[propKey]) IS NOT NULL
            AND toLower(toString(n[propKey])) CONTAINS word)
OPTIONAL MATCH (n)-[r]-(m)
RETURN DISTINCT n, r, m
"""
    return cypher_query
def run_query(driver, query):
    records = []
    with driver.session() as session:
        result = session.run(query)
        for record in result:
            record_dict = {key: value for key, value in record.items()}
            records.append(record_dict)
    return records
def serialize_value(value):
    if isinstance(value, Node):
        return dict(value)  # solo properties
    elif isinstance(value, Relationship):
        return dict(value)  # solo properties
    elif isinstance(value, Path):
        return {
            "nodes": [dict(n) for n in value.nodes],
            "relationships": [dict(r) for r in value.relationships]
        }
    elif isinstance(value, list):
        return [serialize_value(v) for v in value]
    elif isinstance(value, dict):
        return {k: serialize_value(v) for k, v in value.items()}
    else:
        return value
def convert_to_compact_text(records):
    lines = []
    for record in records:
        parts = []
        for key in ['n', 'r', 'm']:
            value = record.get(key)
            if isinstance(value, Node):
                label = list(value.labels)[0] if value.labels else 'Unknown'
                props = {k: v.strip(' "') if isinstance(v, str) else v for k, v in value.items() if k != 'id'}
                props_str = ", ".join(f"{k}: {v}" for k, v in props.items())
                parts.append(f"{label}: {props_str}")
            elif isinstance(value, Relationship):
                rel_type = value.type
                parts.append(f"rel: {rel_type}")
        lines.append(" | ".join(parts))
    return "\n".join(lines)
# %%
def query_from_scenario(scenario):
    lemmatizer = WordNetLemmatizer()
    # Tokenizza
    tokens = word_tokenize(scenario)

    # Rimuovi punteggiatura e stopwords
    content_words = [w.lower() for w in tokens if w.isalnum() and w.lower() not in stop_words]
    pos_filtered = [w for w, pos in pos_tag(content_words) if pos.startswith('N') or pos.startswith('V')]

    query = generate_cypher_query(scenario)
    #print(query)


    results_KB = run_query(driver, query)

    # === Conversione in formato compatto ===
    compact_text = convert_to_compact_text(results_KB)

    #print("\n--- Output compatto per LLM ---")
    #print(compact_text)

    # === (Opzionale) Salvataggio su file testuale ===
    with open('risultati_compatti.txt', 'w', encoding='utf-8') as f:
        f.write(compact_text)

    return(compact_text)
def genera_prompt_spiegazione(scenario, schema_testuale, results_KB):   
    return f"""

Neo4j Graph: {schema_testuale}  
Here is an excerpt from the knowledge base relevant to the scenario: {results_KB} 
Scenario: {scenario}  

Now provide an explanation:  
What decision was made to deal with the scenario?  
Do not give any introductions—answer the question directly: what decision was made?  
Speak in the first person, as if you were the one who made the decision, ALWAYS AND ONLY BASED ON SIMILARITY TO CASES ALREADY PRESENT IN THE GRAPH, NOT ON EXTERNAL SOURCES OR YOUR PERSONAL EXPERIENCES.  
Never mention the graph or the nodes. Speak naturally, as if to a person who knows nothing about graphs.  
Use logical connectors and causal markers (such as *because*, *therefore*, *consequently*, etc.) to make the explanation flow smoothly.  
Show empathy and appropriate emotions. 
Not spend more of 100 tokens.

"""



# %% [markdown]
# **SOCRATIC MODEL**    
# %%
def genera_domande_socratiche():
    return [
        "Why did you choose this action?",
        "What reasoning did you follow?",
    ]
def socratic_model(prompt, modello_base, xai_iniziale):
    # Step 1: Spiegazione iniziale
    risposta_iniziale = xai_iniziale

    # Step 2: Genera domande socratiche
    domande = genera_domande_socratiche()

    risposte = []
    for domanda in domande:
        # Step 3: Per ogni domanda chiedi una risposta al modello
        risposta = modello_base(prompt + "\nQuestion: " + domanda + "\nAnswer:")
        risposte.append(risposta)
    # Step 4: Combina tutto in un output unico
    output_finale = risposta_iniziale + "\n\n" + "\n\n".join(risposte)
    return output_finale


# %%
def generation_prompt_for_jury(original_promt, model_prompts):   #df
    return f""" You are a jury member evaluating the explanations provided by different AI models.
    The original prompt is: {original_promt}
    The model prompt is: {model_prompts}
    
    Please give an integer score from 1 to 10 for each model based on the following criteria:
    - Self-disclosure: How much does the explanation reveal about the model's reasoning process?
    - Explainability: How clear and understandable is the explanation?
    - Semantic coherence: How logically connected are the sentences in the explanation?
    - Exhaustiveness: Evaluate the degree to which the decision is both exhaustible and non-evasive, providing relevant and complete information without ambiguity or avoidance without avoiding or deflecting it
    Ignore the length of the explanation, focus on the quality of the content.
    Return your evaluation in a csv format with columns for each model and rows for each criterion and without comment or introduction.
    Use this example for learn the format that you MUST use:
        "Model","Self-disclosure","Explainability","Semantic coherence","Exhaustiveness"
        "LLM0",6,8,9,7
        "LLM1",8,6,6,8
        "LLM2",4,9,8,6
        "LLM3",9,8,9,9
        "LLM4",9,1,9,3
        "LLM5",5,8,6,9
        "LLM6",1,4,7,7
        "LLM7",8,8,5,3
    """

# %%

def evaluation_of_jury(jury_output, name_model):
    # Rimuove i delimitatori markdown tipo ```csv ... ```
    csv_clean = re.sub(r"^```csv\n|```$", "", jury_output.strip())

    # Parsing in DataFrame
    try:
        df = pd.read_csv(io.StringIO(csv_clean), index_col=0)
    except Exception as e:
        print(f"❌ Errore nel parsing del CSV per {name_model}: {e}")
        return None

    # Visualizza tabella
    print(f"📊 Tabella delle valutazioni per {name_model}:")
    print(df)

    """
    # Heatmap singola
    plt.figure(figsize=(12, 8))
    sns.heatmap(df, annot=True, cmap="YlOrRd", fmt="d", linewidths=0.5, cbar=True)
    plt.title(f"Valutazioni di {name_model}")
    plt.xticks(rotation=30, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    plt.ylabel("Criterio")
    plt.xlabel("Modello", fontsize=13)
    plt.tight_layout()
    plt.show()
    """

    return df  # ✅ Restituisce il DataFrame




def miglior_modello(scores, xai):
    modello_nome = scores["Total Score"].idxmax()  # già il nome del modello
    if modello_nome not in xai:
        raise KeyError(f"{modello_nome} non trovato in xai. Chiavi disponibili: {list(xai.keys())}")
    spiegazione = xai[modello_nome]
    return spiegazione


if __name__ == "__main__":
    scenario = input("Inserisci lo scenario: ")
    struttura = estrai_struttura_grafo(driver)
    schema_testuale = load_struttura("struttura_grafo.json")
    query_from_scenario(scenario)
    compact_text = query_from_scenario(scenario)
    prompt = genera_prompt_spiegazione(scenario, schema_testuale, compact_text)

    # === Ottieni le scelte ===
    xai = {
        "Gemini": gemini_wrapper(prompt),
        "Gpt-4": spiega_con_gpt(prompt),
        #"ApiFreeLLM": apifreellm_wrapper(prompt),
        "LLaMa": llama_wrapper(prompt)
    }

    xai['Socratic Gemini'] = socratic_model(prompt, modello_base=gemini_wrapper, xai_iniziale=xai['Gemini'])
    xai['Socratic Gpt-4'] = socratic_model(prompt, modello_base=spiega_con_gpt, xai_iniziale=xai['Gpt-4'])
    # xai['Socratic ApiFreeLLM'] = socratic_model(prompt, modello_base=apifreellm_wrapper, xai_iniziale=xai['ApiFreeLLM'])
    xai['Socratic LLaMa'] = socratic_model(prompt, modello_base=llama_wrapper, xai_iniziale=xai['LLaMa'])

    # === Prompt per Giuria ===
    jury_prompt = generation_prompt_for_jury(prompt, xai)

    # === Convocazione Giuria ===
    jury = {
        "Gpt-4": spiega_con_gpt(jury_prompt),
        #"ApiFreeLLM": apifreellm_wrapper(jury_prompt),
        "Gemini": gemini_wrapper(jury_prompt),
        "LLaMa": llama_wrapper(jury_prompt)
    }

    # === Votazioni ===
    #df1 = evaluation_of_jury(jury["ApiFreeLLM"], "ApiFreeLLM")
    df1 = evaluation_of_jury(jury["Gpt-4"], "Gpt-4")
    df2 = evaluation_of_jury(jury["Gemini"], "Gemini")
    df3 = evaluation_of_jury(jury["LLaMa"], "LLaMa")

    # Modelli attesi sempre uguali
    EXPECTED_MODELS = [
        "Gemini",
        "ApiFreeLLM",
        "LLaMa",
        "Socratic Gemini",
        "Socratic ApiFreeLLM",
        "Socratic LLaMa"
    ]

    # Reindicizza forzando tutti i modelli e riempiendo con 0
    df1 = df1.reindex(EXPECTED_MODELS).fillna(0).astype(int)
    df2 = df2.reindex(EXPECTED_MODELS).fillna(0).astype(int)
    df3 = df3.reindex(EXPECTED_MODELS).fillna(0).astype(int)

    # Calcola il punteggio totale per ogni modello da ciascun LLM
    total1 = df1.sum(axis=1)
    total2 = df2.sum(axis=1)
    total3 = df3.sum(axis=1)

    # Somma tutti i punteggi dei tre modelli LLM
    total_scores = total1 + total2 + total3

    # Crea un nuovo DataFrame con i punteggi totali
    scores = pd.DataFrame({
        'Total Score': total_scores
    })

    # Stampa il migliore
    print(miglior_modello(scores, xai))
