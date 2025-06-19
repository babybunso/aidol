# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# ### Uncomment the code above in Production Environment
#######################################################################################################

import streamlit as st
from streamlit_feedback import streamlit_feedback
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import openai
from openai import OpenAI
import anthropic
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import spacy
import spacy_streamlit
from wordcloud import WordCloud
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import os
from datetime import datetime
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import regex as re #import re
import json
from annotated_text import annotated_text, parameters

from src.prompts import *

# Development Environment
load_dotenv(override=True)
OPENAI_APIKEY = os.environ['OPENAI_APIKEY']
GEMINI_APIKEY = os.environ['GEMINI_APIKEY']
ANTHROPIC_APIKEY = os.environ['ANTHROPIC_APIKEY']
IS_PROD = False

# # Production Environment
# OPENAI_APIKEY = st.secrets['OPENAI_APIKEY']
# GEMINI_APIKEY = st.secrets['GEMINI_APIKEY']
# ANTHROPIC_APIKEY = st.secrets['ANTHROPIC_APIKEY']
# IS_PROD = True

OPENAI_MODEL = 'gpt-4.1'
ANTHROPIC_MODEL = 'claude-3-5-sonnet-20240620'
GEMINI_MODEL = 'gemini-1.5-pro-latest'
PROJ_DIR = 'lowres/' if IS_PROD else ''
EMBEDDING_MODEL = 'text-embedding-3-large'
SPACY_MODEL = spacy.load(os.path.join(os.getcwd(), f'{PROJ_DIR}en_core_web_sm/en_core_web_sm-3.7.1')) # change to 'en_core_web_lg' if hosted on a server with more resources
ENTITY_LABELS = ['PERSON', 'EVENT', 'DATE', 'GPE', 'ORG', 'FAC', 'LANGUAGE', 'LAW', 'LOC', 'MONEY', 'NORP', 'ORDINAL']
CANDIDATE_LABELS = ['Economic Growth', 'Healthcare Reform', 'Education Initiatives', 'Infrastructure Development', 'Environmental Policies', 'Agricultural Support', 'Employment and Labor', 'Social Welfare Programs', 'Foreign Relations', 'Public Safety and Security']
# SUPPORTED_LANGUAGES_T = ['No translation', 'English', 'Finnish', 'Tagalog', 'Cebuano', 'Ilocano', 'Hiligaynon']
SUPPORTED_LANGUAGES_T = ['English', 'Tagalog', 'Cebuano', 'Ilocano', 'Hiligaynon']
TAGALOG_STOP_WORDS = set("applause nga ug eh yun yan yung kasi ko akin aking ako alin am amin aming ang ano anumang apat at atin ating ay bababa bago bakit bawat bilang dahil dalawa dapat din dito doon gagawin gayunman ginagawa ginawa ginawang gumawa gusto habang hanggang hindi huwag iba ibaba ibabaw ibig ikaw ilagay ilalim ilan inyong isa isang itaas ito iyo iyon iyong ka kahit kailangan kailanman kami kanila kanilang kanino kanya kanyang kapag kapwa karamihan katiyakan katulad kaya kaysa ko kong kulang kumuha kung laban lahat lamang likod lima maaari maaaring maging mahusay makita marami marapat masyado may mayroon mga minsan mismo mula muli na nabanggit naging nagkaroon nais nakita namin napaka narito nasaan ng ngayon ni nila nilang nito niya niyang noon o pa paano pababa paggawa pagitan pagkakaroon pagkatapos palabas pamamagitan panahon pangalawa para paraan pareho pataas pero pumunta pumupunta sa saan sabi sabihin sarili sila sino siya tatlo tayo tulad tungkol una walang ba eh kasi lang mo naman opo po si talaga yung".split())
APP_NAME = 'AIDOL: **Artificial Intelligence Driven Organizer for Language**'
APP_DESC = ' `by GMA Integrated News Team | In partnership with WAN IFRA AI Catalyst`'
ABOUT_AIDOL_1 = """![AIDOL Logo](http://localhost:8501/aidol_logo.png)
**AIDOL** is a smart, AI-powered document intelligence and analysis tool designed for journalists, analysts, and researchers handling large volumes of unstructured content. Developed by the **GMA Integrated News Team**, AIDOL leverages advanced **Large Language Models (LLMs)** and **Natural Language Processing (NLP)** to unlock meaning, relationships, and insights from any data source.

Whether you're working with **text files, PDFs**, or **audio/video inputs** (which AIDOL automatically transcribes), the platform allows users to either:

- Perform **comparative analysis** across **multiple documents or transcripts**, surfacing semantic similarities, contradictions, and narrative shifts  
- Or conduct a **dedicated, in-depth analysis** on a **single input**, extracting key points, sentiment, named entities, and themes

**Ideal for newsroom use cases**, AIDOL enhances editorial workflows by rapidly surfacing patterns, key quotes, and contextual connections — especially in press conferences, hearings, or long-form investigative material.

---

### 🔑 Key Features

- 🔍 **Comparative Document Analysis** – Detects overlaps, contradictions, and nuanced differences  
- 🧠 **Insightful Single-Document Summaries** – Breaks down long content into actionable insights  
- 🎙️ **Speech-to-Text Integration** – Automatically transcribes audio or video into clean, analyzable text  
- 📂 **Cross-Format Support** – Works with PDFs, TXT, and media files  
- 📈 **Self-Improving Engine** – Learns and adapts over time to improve accuracy

---"""
ABOUT_AIDOL_2 = """**AIDOL** empowers teams to go beyond surface-level analysis and uncover the deeper story hidden in the data — faster, smarter, and with more confidence."""
K = 10
DEFAULT_NUM_INPUT = 10
MAX_DOCS = 5
EXP_MAX_DOCS = 30
MIN_RESULTS_PER_DOC = 5
MAX_RESULTS_PER_DOC = 50
MIN_NUM_KWORDS = 5
MAX_NUM_KWORDS = 20
COLLECTION_NAME = "aidol"
APP_DOCS = os.path.join(os.getcwd(), f'{PROJ_DIR}data/documents')
DF_CSV = os.path.join(os.getcwd(), f'{PROJ_DIR}data/aidol.csv')
DB_PATH = os.path.join(os.getcwd(), f'{PROJ_DIR}data/aidol.db')
FEEDBACK_FILE = os.path.join(os.getcwd(), f'{PROJ_DIR}data/feedback.csv')
FEEDBACK_FACES = {
    "😀": ":smiley:", "🙂": ":sweat_smile:", "😐": ":neutral_face:", "🙁": ":worried:", "😞": ":disappointed:"
}

COLOR_BLUE = "#0c2e86"
COLOR_RED = "#a73c07"
COLOR_YELLOW = "#ffcd34"
COLOR_GRAY = '#f8f8f8'

SCROLL_BACK_TO_TOP_BTN = f"""
<style>
    .scroll-btn {{
        position: absolute;
        border: 2px solid #31333f;
        background: #31333f;
        border-radius: 10px;
        padding: 2px 10px;
        bottom: 0;
        right: 0;
    }}

    .scroll-btn:hover {{
        color: #ff4b4b;
        border-color: #ff4b4b;
    }}
</style>
<a href="#start">
    <br />
    <button class='scroll-btn'>
        New Query
    </button>
</a>
"""

SAFETY_SETTINGS = [
    {
        "category": "HARM_CATEGORY_DANGEROUS",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]

NEWSROOM_HS = "Helsingin Sanomat"
DEFAULT_NUM_TOPIC_LABELS = 3
MIN_TOPIC_LABELS = 3
MAX_TOPIC_LABELS = 10
MAX_KEYWORDS = 10
MIN_SENT_WORD_COUNT = 10
MAX_SENT_COUNT = 50
TOPIC_FILE = os.path.join(os.getcwd(), f'{PROJ_DIR}data/topics.csv')
#######################################################################################################

def get_openai_client():
    client = OpenAI(api_key=OPENAI_APIKEY)
    return client
#######################################################################################################

def get_anthropic_client():
    client = anthropic.Client(api_key=ANTHROPIC_APIKEY)
    return client
#######################################################################################################

def get_gemini_client():
    genai.configure(api_key=GEMINI_APIKEY)
    return genai.GenerativeModel(GEMINI_MODEL)
#######################################################################################################

def init_chroma_db(collection_name, db_path=DB_PATH):
    # Create a Chroma Client
    chroma_client = chromadb.PersistentClient(path=db_path)

    # Create an embedding function
    embedding_function = OpenAIEmbeddingFunction(api_key=OPENAI_APIKEY, model_name=EMBEDDING_MODEL)

    # Create a collection
    collection = chroma_client.get_or_create_collection(name=collection_name, embedding_function=embedding_function)

    return collection
#######################################################################################################

# Function to chunk the documents before upserting them
def chunk_document(text, chunk_size=1000, chunk_overlap=100):
    # Use a RecursiveCharacterTextSplitter for chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_text(text)
#######################################################################################################

def get_documents_from_vdb(collection=None, title=None):
    return collection.get(
        where={'title': title}
    )
#######################################################################################################

def semantic_search(Q, k=K, collection=None,  titles=[]):
    n_K = len(titles) * k
    results = collection.query(
        query_texts=[Q], # Chroma will embed this for you
        n_results=n_K, # how many results to return,
        where={ 'title': {'$in': titles} }
    )
    return results

# def semantic_search(Q, k=5, collection=None):
#     # Query the collection
#     results = collection.query(
#         query_texts=[Q], # Chroma will embed this for you
#         n_results=k # how many results to return
#     )
#     return results
#######################################################################################################

def semantic_search_separated_documents(Q, k=K, collection=None, titles=[]):
    results = {
        'ids': [[]],
        'distances': [[]],
        'metadatas': [[]],
        'documents': [[]]
    }
    for title in titles:
        title_results = collection.query(
            query_texts=[Q],
            n_results=k,
            where={'title': title}
        )
        for key in results:
            if key in title_results and isinstance(results[key], list) and isinstance(title_results[key], list):
                results[key][0].extend(title_results[key][0])
    return results
#######################################################################################################

def semantic_search_expanded(Q, expanded_queries, k=K, collection=None, titles=[], separate_documents=False, llm=None):
    expanded_queries.append(Q)

    results = {
        'ids': [[]],
        'distances': [[]],
        'metadatas': [[]],
        'documents': [[]]
    }

    for query in expanded_queries:
        if separate_documents:
            partial_results = semantic_search_separated_documents(query, k, collection, titles)
        else:
            partial_results = semantic_search(query, k, collection, titles)

        for key in results:
            if key in partial_results and isinstance(results[key], list) and isinstance(partial_results[key], list):
                results[key][0].extend(partial_results[key][0])

    # Remove duplicates from documents and corresponding metadata, distances, and ids
    seen_documents = set()
    unique_results = {
        'ids': [[]],
        'distances': [[]],
        'metadatas': [[]],
        'documents': [[]]
    }

    for i, doc in enumerate(results['documents'][0]):
        if doc not in seen_documents:
            seen_documents.add(doc)
            unique_results['documents'][0].append(doc)
            unique_results['ids'][0].append(results['ids'][0][i])
            unique_results['distances'][0].append(results['distances'][0][i])
            unique_results['metadatas'][0].append(results['metadatas'][0][i])
            
    unique_results['distances'][0], unique_results['documents'][0] = zip(*sorted(zip(unique_results['distances'][0], unique_results['documents'][0])))

    return unique_results
#######################################################################################################

def upsert_documents_to_collection(collection, documents, chunk_size, chunk_overlap):
    # Every document needs an id for Chroma
    last_idx = len(collection.get()['ids'])
    all_ids = []
    all_documents = []
    all_metadatas = []

    for idx, doc in enumerate(documents):
        # Split each document into chunks
        chunks = chunk_document(doc.page_content, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        # Assign a unique ID to each chunk
        ids = [f'id_{idx+last_idx:010d}_chunk_{i}' for i in range(len(chunks))]
        
        # Append chunked documents and their metadata
        all_ids.extend(ids)
        all_documents.extend(chunks)
        all_metadatas.extend([doc.metadata] * len(chunks))  # Metadata is repeated for each chunk

    # Upsert the chunked documents into the ChromaDB collection
    collection.upsert(ids=all_ids, documents=all_documents, metadatas=all_metadatas)
#######################################################################################################

def expand_query(Q, newsroom, doc_lang, nr_queries=4, llm=None, temperature=1):
    task = 'Query Expansion'
    prompt = QUERY_EXPANSION_PROMPT.replace('{{NR_QUERIES}}', str(nr_queries)).replace('{{Q}}', Q)
    if newsroom == NEWSROOM_HS and doc_lang == 'No translation':
        prompt = prompt.replace('{{LANGUAGE}}', 'Answer in Finnish.')
    else:
        prompt = prompt.replace('{{LANGUAGE}}', '')
    response = generate_response(task, prompt, llm, temperature)
    expanded_queries = response.split('\n')  # Assuming each variation is on a new line
    expanded_queries = [query.strip() for query in expanded_queries]
    filtered_queries = [s for s in expanded_queries if s and len(s) <= 500]
    return filtered_queries
#######################################################################################################

def generate_response(task, prompt, llm, temperature=0.):
    if isinstance(llm, OpenAI):
        response = llm.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {'role': 'system', 'content': f"Perform the specified task: {task}"},
                {'role': 'user', 'content': prompt}
            ],
            temperature=temperature
        )
        return response.choices[0].message.content
    
    elif isinstance(llm, anthropic.Client):
        response = llm.messages.create(
            max_tokens=5000,
            messages=[
                {'role': 'user', 'content': prompt}
            ],
            model=ANTHROPIC_MODEL,
            temperature=temperature,
        )
        return response.content[0].text
    
    elif isinstance(llm, genai.GenerativeModel):
        response = llm.generate_content(
            prompt,
            safety_settings=SAFETY_SETTINGS,
            generation_config=genai.types.GenerationConfig(temperature=temperature)
        )
        return response.text
    
    return None
#######################################################################################################

def generate_summarization(doc, llm):
    task = 'Text Summarization'
    prompt = f"Summarize this document:\n\n{doc}"
    response = generate_response(task, prompt, llm)
    return response
#######################################################################################################

def generate_focused_summarization(Q, doc, llm, newsroom, doc_lang):
    task = 'Text Summarization'
    prompt = FOCUSED_SUMMARIZATION_PROMPT.replace('{{QUESTION}}', Q).replace('{{DOCUMENT}}', doc)
    if newsroom == NEWSROOM_HS and doc_lang == 'No translation':
        prompt = prompt.replace('{{LANGUAGE}}', 'Answer in Finnish.')
    else:
        prompt = prompt.replace('{{LANGUAGE}}', '')
    response = generate_response(task, prompt, llm)
    return response
#######################################################################################################

def generate_translation(doc, target_lang, llm):
    response = doc
    if target_lang != 'English' or target_lang != 'No translation':
        task = 'Text Translation'
        prompt = f"Translate this document to {target_lang}:\n\n{doc}\n\n\nOnly respond with the translation."
        response = generate_response(task, prompt, llm)
    return response
#######################################################################################################

def generate_keyword_labels(doc, llm, top_k=K):
    task = 'Topic Modeling or keyword extraction'
    prompt = f"Extract and list the top {top_k} main keywords in this document:\n\n{doc}"
    response = generate_response(task, prompt, llm)
    return response
#######################################################################################################

def generate_sentiment_analysis(doc, llm):
    task = 'Sentiment Analysis'
    prompt = f"Classify the sentiment analysis of this document:\n\n{doc}\n\n\n Use labels: Positive, Negative, Neutral, Mixed"
    response = generate_response(task, prompt, llm)
    return response
#######################################################################################################

def generate_document_analysis(Q, df, llm, advanced_prompt):
    task = 'Document analysis and comparison'
    titles = df['title'].to_list()
    documents = df['documents'].to_list()
    doc_input = ''
    for i in range(len(df)):
        doc_input += f"""
        Document {i} Title: {titles[i]}
        Document {i} Content: {documents[i]}
        """
    title_0 = titles[0]
    title_1 = titles[0]
    try:
        title_1 = titles[1]
    except:
        pass
    prompt = advanced_prompt.replace('{{QUESTION}}', Q).replace('{{DOCUMENTS}}', doc_input).replace('{{TITLE_0}}', title_0).replace('{{TITLE_1}}', title_1)
    response = generate_response(task, prompt, llm)
    return response
#######################################################################################################

def generate_document_analysis_hs(Q, titles, texts, llm, advanced_prompt, doc_lang): # HS_ANALYSIS
    task = 'Document analysis and comparison'
    doc_input = 'Each document has a title and content and is delimited by triple backticks.'
    for i in range(len(titles)):
        doc_input += f"""
        ```Document title: {titles[i]} Content: {texts[i]}```
        """

    prompt = advanced_prompt.replace('{{QUESTION}}', Q).replace('{{DOCUMENTS}}', doc_input)
    if doc_lang == 'No translation':
        prompt = prompt.replace('{{LANGUAGE}}', 'Answer in Finnish.')
    response = generate_response(task, prompt, llm)
    return response
#######################################################################################################

def generate_response_to_question(Q, text, titles, llm):
    title_0 = titles[0]
    title_1 = titles[0]
    try:
        title_1 = titles[1]
    except:
        pass
    """Generalized function to answer a question"""
    prompt = f"""
    Provide the answer on {Q} based on {', '.join(titles)} given this document:\n\n{text}.

    You should only respond based on the given documents. if you don't know the answer, just respond you don't know the answer. Don't give more than what is asked. Only answer the questions directly related to the {', '.join(titles)} and the given report. If not directly stated in the report, say that and don't give assumptions.

    For the answer, you would include a reference to a phrase where you have found the answer. e.g. "Source: Document 0 Title: {titles[0]}" or "Sources: Document 0 and 1; Titles: {title_0} and {title_1}".
    """
    response = generate_response(Q, prompt, llm)
    return response
#######################################################################################################

def ask_query(Q, titles, llm, collection, k=15):
    """Function to go from question to query to proper answer"""
    # Get related documents
    results = semantic_search(Q, k=K, collection=collection, titles=titles)

    # Get the text of the documents
    # text = query_result['documents'][0][0] TODO
    text = ''
    for t in results['documents'][0]:
        text += t

    # Pass into GPT to get a better formatted response to the question
    response = generate_response_to_question(Q, text, titles, llm=llm)
    # return Markdown(response)
    return response
#######################################################################################################

def plot_wordcloud(df, column):
    # Data with filled of additonal stop words
    my_stop_words = list(text.ENGLISH_STOP_WORDS.union(TAGALOG_STOP_WORDS))

    # Fit vectorizers
    count_vectorizer = CountVectorizer(stop_words=my_stop_words)
    cv_matrix = count_vectorizer.fit_transform(df[column])

    tfidf_vectorizer = TfidfVectorizer(stop_words=my_stop_words)
    tfidf_matrix = tfidf_vectorizer.fit_transform(df[column])

    # Create dictionaries for word cloud
    count_dict = dict(zip(count_vectorizer.get_feature_names_out(),
                                cv_matrix.toarray().sum(axis=0)))

    tfidf_dict = dict(zip(tfidf_vectorizer.get_feature_names_out(),
                                tfidf_matrix.toarray().sum(axis=0)))

    # Create word cloud and word frequency visualization
    count_wordcloud = (WordCloud(width=800, height=400, background_color='black')
                    .generate_from_frequencies(count_dict))

    tfidf_wordcloud = (WordCloud(width=800, height=400, background_color='black')
                    .generate_from_frequencies(tfidf_dict))

    # Plot the word clouds and word frequency visualizations
    fig = plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(count_wordcloud, interpolation='bilinear')
    plt.title('Count Vectorizer Word Cloud')
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(tfidf_wordcloud, interpolation='bilinear')
    plt.title('TF-IDF Vectorizer Word Cloud')
    plt.axis("off")

    plt.tight_layout()
    plt.show();
    return fig
#######################################################################################################

# @st.cache_data()
def init_data():
    df = pd.DataFrame(columns=['url', 'title', 'speech'])
    try:
        df = pd.read_csv(DF_CSV)
    except:
        pass
    return df
#######################################################################################################

def save_uploadedfile(uploadedfile):
    if not os.path.exists(APP_DOCS):
        os.makedirs(APP_DOCS)
    file_path = uploadedfile.name

    with open(os.path.join(APP_DOCS, file_path), "wb") as f:
        f.write(uploadedfile.getbuffer())
    return file_path
#######################################################################################################

def submit_feedback(feedback, *args, **kwargs):
    score = feedback['score']
    reax = FEEDBACK_FACES.get(score)
    newsroom = kwargs["newsroom"]
    fb_type = kwargs['type']
    question = kwargs['question']
    prompt = kwargs['prompt']
    llm_response = kwargs['llm_response']
    documents = kwargs['documents']
    feedback_sent = kwargs["feedback_time"]
    comment = feedback['text']

    fbk_from = kwargs['fbk_from']
    if fbk_from == 'rag_chatbot_fbk':
        st.session_state['ragchat_feedback_sent'] = True
    elif fbk_from == 'document_analyzer_fbk':
        st.session_state['docanalyzer_feedback_sent'] = True
    try:
        fb_data = {
            'timestamp': feedback_sent,
            'type': fb_type,
            'newsroom': newsroom,
            'documents': documents,
            'question': question,
            'prompt': prompt,
            'response': llm_response,
            'reaction': reax,
            'score': score,
            'comment': comment
        }
        new_df = pd.DataFrame([fb_data])
        if os.path.exists(FEEDBACK_FILE):
            fb_df = pd.read_csv(FEEDBACK_FILE)
            df = pd.concat([fb_df, new_df], axis=0).reset_index(drop=True)
            df.to_csv(FEEDBACK_FILE, index=False)
        else:
            new_df.to_csv(FEEDBACK_FILE, index=False)

    except Exception as ex:
        st.error(ex)
#######################################################################################################

def display_feedback():
    if os.path.exists(FEEDBACK_FILE):
        st.dataframe(pd.read_csv(FEEDBACK_FILE), height=750, width=1400)
    else:
        st.error('No feedback submitted yet.')
#######################################################################################################

def scroll_to_top():
    js = '''
    <script>
        var body = window.parent.document.querySelector("#start");
        console.log(body);
        body.scrollTop = 0;
    </script>
    '''
    st.html(js)
#######################################################################################################

def set_page_icon(icon_label='🏠'):
    c_op1, c_op2 = st.columns([1, 18])
    c_op1.subheader(icon_label)
    c_op2.write('___')
#######################################################################################################

def reset_document_analyzer():
    for key in st.session_state.keys():
        if 'newsroom' != key:
            del st.session_state[key]
#######################################################################################################

def reset_ragchatbot():
    reset_document_analyzer()
#######################################################################################################

def blank(container=None, lines=1):
    for _ in range(lines):
        if container:
            container.markdown('<div></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div></div>', unsafe_allow_html=True)
#######################################################################################################

@st.dialog("What newsroom are you affiliated with?")
def set_newsroom():
    newsrooms = ['Helsingin Sanomat', 'GMA Network']
    newsroom = st.radio("Select Newsroom:", newsrooms)
    # name = st.text_input("Enter your name:")
    # position = st.text_input("Enter your position:")
    st.container(height=10, border=0)
    if st.button("Select"):
        st.session_state.newsroom = {"newsroom": newsroom} #, "name": name, "position": position}
        st.rerun()
#######################################################################################################

def set_docanalyzer_feedback(Q, prompt, document_analysis, QDOCS, newsroom):
    if 'docanalyzer_feedback_sent' not in st.session_state:
        st.session_state.docanalyzer_feedback_sent = False
    current_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    fbk_from = 'document_analyzer_fbk'
    if streamlit_feedback(
        feedback_type="faces",
        optional_text_label="Please provide some more information here...",
        align='flex-start',
        kwargs={"fbk_from":fbk_from, "type": "document_analyzer", "newsroom": newsroom, "question": Q, "prompt": prompt, "llm_response": document_analysis, "documents": ', '.join(QDOCS), "feedback_time": current_date},
        on_submit=submit_feedback
    ):
        st.session_state.docanalyzer_feedback_sent = True
#######################################################################################################

def generate_topic_labels(doc, llm, n_topics=DEFAULT_NUM_TOPIC_LABELS):
    task = 'Topic Modeling'
    prompt = TOPIC_MODELING_PER_DOC_PROMPT.replace('{{DOCUMENT}}', doc).replace('{{TOPIC_COUNT}}', str(n_topics))
    response = generate_response(task, prompt, llm)
    return response
#######################################################################################################

def combine_topic_labels(candicate_topics, llm):
    task = 'Topic Modeling'
    prompt = COMBINE_TOPIC_LABELS_PROMPT.replace('{{TOPICS}}', '; '.join(candicate_topics))
    response = generate_response(task, prompt, llm)
    return response
#######################################################################################################

def generate_topics(docs_df, llm, n_topics):
    topics = []
    for i, doc in enumerate(docs_df['document chunks']):
        try:
            response = generate_topic_labels(doc, llm, n_topics)
            if response:
                response = response.replace('```json', '').replace('```', '')
                response = json.loads(response)
                # response = eval(response)
                data_dict = {'chunk_id': docs_df['ids'].iloc[i], 'topics': response}
                topics.append(data_dict)
        except:
            pass
    return topics
#######################################################################################################

def combine_topics(l_topics, llm, col_label='topics'):
    try:
        topics = []
        for candicate_topics in l_topics:
            topics.extend(candicate_topics[col_label])
        topics = list(set(topics))

        response = combine_topic_labels(topics, llm)
        response = response.replace('```json', '').replace('```', '')
        response = json.loads(response)
        # response = eval(response)
        return topics, response
    except:
        st.error('Error combining topics. Please try again.')
#######################################################################################################

def generate_annotation(doc_texts, topics, llm):
    task = 'Text Annotation'
    prompt = DOC_ANNOTATION_PROMPT.replace('{{DOCUMENT}}', doc_texts).replace('{{TOPICS}}', ', '.join(topics)).replace('{{MAX_SENT_COUNT}}', str(MAX_SENT_COUNT)).replace('{{MIN_SENT_WORD_COUNT}}', str(MIN_SENT_WORD_COUNT)).replace('{{TOPIC_0}}', topics[0]).replace('{{TOPIC_1}}', topics[1])
    response = generate_response(task, prompt, llm)
    return response
#######################################################################################################

def get_annotations(docs_df, topics, llm):
    annot_texts = []
    for doc in docs_df['document chunks']:
        response = generate_annotation(doc, topics, llm)
        annot_texts.append(response)
    return annot_texts
#######################################################################################################

def get_annotated_texts(annot_results, combined_topics):
    all_annot_texts = []
    for annot in annot_results:
        annot_all_topics = [f'\[{s}\].*?\[\/{s}\]' for s in combined_topics]
        annot_all_topics_regex_f = '|'.join(annot_all_topics)
        regex_all_res = re.findall(annot_all_topics_regex_f, annot, re.DOTALL)

        annot_texts = []
        for ret in regex_all_res:
            annot = None
            if ret is None or (not isinstance(ret, str)):
                continue
            selected_topic = re.findall('\[\/.*?\]', ret, re.DOTALL)
            if selected_topic and len(selected_topic)>0:
                selected_topic = selected_topic[0].replace('[/', '').replace(']', '')
                new_atext = ret.replace(f'[/{selected_topic}]', '').replace(f'[{selected_topic}]', '')
                if selected_topic == 'NO_TOPIC':
                    annot_texts.append(new_atext)
                else:
                    annot_texts.append((new_atext, selected_topic.upper()))
        all_annot_texts.append(annot_texts)
    return [all_annot_texts]
#######################################################################################################

def save_topic_modeling_results(QDOC, combined_topics, recommended_topics_and_subtopics, topics_per_doc_chunk, all_annot_texts):
    try:
        topic_data = {
            'document': QDOC,
            'combined_topics': str(combined_topics),
            'recommended_topics_and_subtopics': str(recommended_topics_and_subtopics),
            'topics_per_doc_chunk': str(topics_per_doc_chunk),
            'annotated_texts': str(all_annot_texts)
        }

        df = None
        new_topics_df = pd.DataFrame([topic_data])
        if os.path.exists(TOPIC_FILE):
            topics_df = pd.read_csv(TOPIC_FILE)
            if len(topics_df[topics_df['document']==QDOC]) > 0:
                topics_df.loc[topics_df['document']==QDOC] = new_topics_df.values[0]
                df = topics_df
            else:
                df = pd.concat([topics_df, new_topics_df], axis=0).reset_index(drop=True)
        else:
            df = new_topics_df

        df.to_csv(TOPIC_FILE, index=False)
    except Exception as ex:
        st.error(ex)
#######################################################################################################

def get_preprocessed_topic_df():
    if os.path.exists(TOPIC_FILE):
        topics_df = pd.read_csv(TOPIC_FILE)
    else:
        topics_df = pd.DataFrame(columns=['document', 'combined_topics', 'recommended_topics_and_subtopics', 'topics_per_doc_chunk', 'annotated_texts'])
    return topics_df

def get_preprocessed_doc_titles():
    topics_df = get_preprocessed_topic_df()
    return topics_df['document'].unique().tolist()

def get_topic_df(QDOC):
    topics_df = get_preprocessed_topic_df()
    return topics_df[topics_df['document']==QDOC]

def check_topics_in_docs(QDOCS, container):
    for qdoc in QDOCS:
        topic_df = get_topic_df(qdoc)
        if topic_df is not None and len(topic_df) > 0:
            existing_topics = eval(topic_df['combined_topics'].iloc[0])
            annot_existing_labels = [(str(i), t.upper()) for i, t in enumerate(existing_topics)]
            with container:
                annotated_text(annot_existing_labels)
                st.caption(f"This document already has recommended topics. Click 'Generate Topics' to update the topics or select another document.")
            # chk_preview = st.checkbox('Preview Annotated Document', value=False, key='chk_preview')
            # if chk_preview:
            #     annot_texts = eval(topic_df['annotated_texts'].iloc[0])
            #     annotated_text(annot_texts)

