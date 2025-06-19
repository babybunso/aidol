from src.utils import *
#######################################################################################################
#######################################################################################################

# Create a Streamlit app
st.set_page_config(layout="wide", page_icon='📰', page_title=APP_NAME)
st.markdown('<div id="start"></div>', unsafe_allow_html=True)
st.title(APP_NAME)
st.write(APP_DESC)

# Initialize newsroom selection
if "newsroom" not in st.session_state:
    set_newsroom()

# Initialize chroma db
collection = init_chroma_db(collection_name=COLLECTION_NAME, db_path=DB_PATH)

# Load the dataset
df = init_data()

if "newsroom" in st.session_state:
    newsroom = st.session_state.newsroom["newsroom"]
    st.sidebar.markdown(f"Newsroom:  &nbsp; **{newsroom}**")
    st.sidebar.button(":newspaper: &nbsp; Change", on_click=set_newsroom)
    st.sidebar.write('___')
    
chosen_model = st.sidebar.radio("Choose model:", ["OpenAI GPT-4x", "Anthropic Claude 3.5 Sonnet", "Gemini 1.5 Pro"])
st.sidebar.write('___')

# Initialize OpenAI client
if chosen_model == "OpenAI GPT-4x":
    llm = get_openai_client()
elif chosen_model == "Anthropic Claude 3.5 Sonnet":
    llm = get_anthropic_client()
elif chosen_model == "Gemini 1.5 Pro":
    llm = get_gemini_client()
    
st.sidebar.container(height=20, border=0)
options = st.sidebar.radio("Menu", ["🏠 Home", "📊 The Dataset", "🏷️ Topic Modeling", "📚 Document Analysis", "🎫 Feedback Page"], label_visibility='hidden')
st.sidebar.container(height=40, border=0)
st.sidebar.write('___')


if options == "🏠 Home":
    set_page_icon('🏠')
    st.subheader("About")

    st.write(ABOUT_AIDOL_1)
    st.write(ABOUT_AIDOL_2)
    st.write('___')
    st.caption("For more information, visit our GitHub repository: [AIDOL](https://github.com/journalismAI-aidol/aidol)")
    reset_document_analyzer()

if options == "📊 The Dataset":
    set_page_icon('📊')
    st.write("##### Upload a document to add to the dataset:")
    c1, c2 = st.columns([1, 1])
    chunk_size = c1.number_input('Chunk Size:', min_value=100, max_value=3000, value=1000, step=100)
    chunk_overlap = c2.number_input('Chunk Overlap:', min_value=0, max_value=500, value=100, step=10)
    pdfs = st.file_uploader('Upload PDF files', accept_multiple_files=True, type=['pdf'], label_visibility='hidden')
    btn_upload = st.button("Upload", type='primary')
    if pdfs and btn_upload:
        for pdf in pdfs:
            file_path = save_uploadedfile(pdf)
            st.toast(f"File uploaded successfully: {file_path}")
            loader = PyPDFLoader(f'{APP_DOCS}/{file_path}')
            docs = loader.load()

            metadata = {'url':f'file://{file_path}', 'title':file_path}

            # for dataframe
            doc_input = ''
            for i_tmp, doc in enumerate(docs):
                doc_input += str(doc.page_content)
                doc.metadata = metadata
            upsert_documents_to_collection(collection, docs, chunk_size, chunk_overlap) # NOTE: Run Once to insert the documents to the vector database
            new_df = pd.DataFrame([{'url': f'file://{file_path}', 'title': file_path, 'speech': doc_input}])
            df = init_data()
            df = pd.concat([df, new_df], axis=0).reset_index(drop=True)
            df.to_csv(DF_CSV, index=False)

    st.write("___")
    df = init_data()
    c1, c2 = st.columns([2, 1])
    c2.subheader("Word Count:")
    c2.write(f"{df['speech'].apply(lambda x: len(x.split())).sum(): ,}")
    c1.subheader("The Dataset:")
    c1.write(f"The dataset contains {len(df)} documents.")
    display_df = df.rename(columns={'speech': 'content'}).copy()
    st.dataframe(display_df, height=750, width=1400)
    reset_document_analyzer()

if options == "🏷️ Topic Modeling":
    set_page_icon('🏷️')

    QDOC = ''
    cf1, _, cf2 = st.columns([20, 1, 5])
    with cf1:
        QDOC = st.selectbox("Select a Document for Topic Modeling Task:", df['title'].unique(), index=None, placeholder="Select a document...")
        topic_df = get_topic_df(QDOC)
        if topic_df is not None and len(topic_df) > 0:
            existing_topics = eval(topic_df['combined_topics'].iloc[0])
            annot_existing_labels = [(str(i), t.upper()) for i, t in enumerate(existing_topics)]
            annotated_text(annot_existing_labels)
            st.caption(f"This document already has recommended topics. Click 'Generate Topics' to update the topics or select another document.")
            chk_preview = st.checkbox('Preview Annotated Document', value=False, key='chk_preview')
            if chk_preview:
                annot_texts = eval(topic_df['annotated_texts'].iloc[0])
                annotated_text(annot_texts)

    with cf2:
        blank()
        st.write("###### Topic Modeling Options:")

        blank()
        n_topics = st.number_input('No. of Topics to generate per chunk:', min_value=MIN_TOPIC_LABELS, max_value=MAX_TOPIC_LABELS, value=DEFAULT_NUM_TOPIC_LABELS, step=1)
        blank()
        chk_annotate = st.checkbox('Annotate Document', value=False, key='chk_annotate')
        st.caption("__NOTE:__ This post-process would take some time for long documents.")
        blank()

    if QDOC and QDOC != '':
        result_docs = get_documents_from_vdb(collection, QDOC)
        docs_df = pd.DataFrame(result_docs, columns=['ids', 'documents', 'metadatas'])
        docs_df.rename(columns={'documents': 'document chunks'}, inplace=True)
        blank(cf1, 2)
        cf1.write("Document Information:")
        cf1.caption(f'Number of Chunks:__&nbsp;&nbsp;&nbsp; {len(docs_df)}__')
        cf1.caption(f'Word Count:__&nbsp;&nbsp;&nbsp; {docs_df["document chunks"].apply(lambda x: len(x.split())).sum(): ,}__')
        cf1.dataframe(docs_df, height=400, use_container_width=True)
        blank(cf1, 2)

        btn_generate = cf1.button("Generate Topics", type='primary')

        if btn_generate:
            results1 = generate_topics(docs_df, llm, n_topics)
            topics_per_doc_chunk, results2 = combine_topics(results1, llm)
            if results2 is None:
                st.error("Topic Modeling failed. Please try again.")
            else:
                combined_topics = [item['topic'] for item in results2]

                container = st.container()
                container.write('___')
                container.subheader("Topic Modeling Results")
                container.caption(f"Total Number of Recommended Topic Labels:__&nbsp;&nbsp;&nbsp; {len(results2)}__")
                container.caption(f"Total Number of Unique Sub-topic Labels based on Document Chunks:__&nbsp;&nbsp;&nbsp; {len(topics_per_doc_chunk)}__")
                container.caption(f'Topic Labels:')
                with container:
                    annot_labels = [(str(i), t.upper()) for i, t in enumerate(combined_topics)]
                    annotated_text(annot_labels)
                blank(container, 3)
                cr1, cr2 = container.columns([1, 1])
                cr2.markdown("##### Topics per Document Chunk:")
                cr2.write(results1)
                cr1.markdown("##### Recommended Topics:")
                cr1.write(results2)
                blank(container, 3)
                all_annot_texts = ['- ANNOTATION IS NOT AVAILABLE FOR THIS DOCUMENT -']
                if chk_annotate:
                    container.markdown('#### Annotated Document Preview:')
                    annot_results = get_annotations(docs_df, combined_topics, llm)
                    all_annot_texts = get_annotated_texts(annot_results, combined_topics)
                    with container:
                        annotated_text(all_annot_texts)
                        container.write('___')

                # save the topic modeling results
                save_topic_modeling_results(QDOC, combined_topics, results1, results2, all_annot_texts)
                st.toast('Topic Modeling results saved successfully!', icon='🎉')
    reset_document_analyzer()

if options == "🎫 Feedback Page":
    set_page_icon('🎫')
    display_feedback()
    reset_document_analyzer()

if options == "📚 Document Analysis":
    set_page_icon('📚')
    newsroom = ''
    if 'newsroom' in st.session_state:
        newsroom = st.session_state.newsroom["newsroom"]
    st.sidebar.container(height=5, border=0)
    st.sidebar.caption(':gear: &nbsp;&nbsp; Experimental Options for "Document Analyzer":')
    if 'num_experimental_max_docs' not in st.session_state:
        st.session_state['num_experimental_max_docs'] = MAX_DOCS
    st.sidebar.number_input('Maximum Number of Selected Documents', min_value=MAX_DOCS, max_value=EXP_MAX_DOCS, step=1, key='num_experimental_max_docs')

    if st.session_state['num_experimental_max_docs'] > MAX_DOCS:
        st.sidebar.warning(f'**Note:** Output options such as _Text Summarization with Translation_, _Sentiment Classification_, _Keyword Extraction_, and _Name Entity Recognition (NER)_  is currently not supported if the number documents to be selected is more than {MAX_DOCS}.')

    st.sidebar.container(height=5, border=0)
    if 'chk_advanced_prompt' not in st.session_state:
        st.session_state['chk_advanced_prompt'] = False
    st.sidebar.checkbox('Show Advanced Prompt Engineering', value=st.session_state['chk_advanced_prompt'], key='chk_advanced_prompt')
    prompt_template = ''
    if newsroom == NEWSROOM_HS:
        if st.session_state['num_experimental_max_docs'] > MAX_DOCS:
            prompt_template = HS_PROMPT_MANY_DOCUMENTS
        else:
            prompt_template = HS_PROMPT_FINNISH
    else:
        prompt_template = DOC_ANALYSIS_BASE_PROMPT

    advanced_prompt = ''
    if st.session_state['chk_advanced_prompt']:
        if newsroom == NEWSROOM_HS:
            st.sidebar.warning('**Note:** _{{QUESTION}}_, and _{{DOCUMENTS}}_ are placeholders for the actual values.')
        else:
            st.sidebar.warning('**Note:** _{{QUESTION}}_, _{{DOCUMENTS}}_, _{{TITLE_0}}_, and _{{TITLE_1}}_ are placeholders for the actual values.')
        advanced_prompt = st.sidebar.text_area("Prompt Template:", placeholder="Type your prompt here...", value=prompt_template.strip(), height=800, max_chars=5000)
    else:
        advanced_prompt = prompt_template

    if len(df) < 1:
        st.error('Please upload at least one document in the "📊 The Dataset" page to start the document analysis.')
    else:
        tab1, tab2 = st.tabs(["Document Analyser", "RAG ChatBot"])

        # Document Analyzer
        with tab1:
            Q = ''
            QA = '' # will clean this up later
            QT = []
            QDOCS = []
            qa_val = ''
            mselect_qt = []
            mselect_qdocs = []
            if 'key_txt_qa' in st.session_state:
                qa_val = st.session_state['key_txt_qa'] # will clean this up later
            if 'key_mselect_qt' in st.session_state:
                mselect_qt = st.session_state['key_mselect_qt']
            if 'key_mselect_qdocs' in st.session_state:
                mselect_qdocs = st.session_state['key_mselect_qdocs']

            cf1, _, cf2 = st.columns([11, 1, 4])
            with cf1:
                if 'chk_preprocessed_topic' not in st.session_state:
                    st.session_state['chk_preprocessed_topic'] = False
                st.checkbox('Filter Documents by Preprocessed Topics', value=st.session_state['chk_preprocessed_topic'], key='chk_preprocessed_topic')

                document_titles = None
                if st.session_state['chk_preprocessed_topic']:
                    preprocessed_titles = get_preprocessed_doc_titles()
                    document_titles = df[df['title'].isin(preprocessed_titles)]
                    document_titles = document_titles['title'].unique()
                else:
                    document_titles = df['title'].unique()

                QDOCS = st.multiselect("Select Documents:", document_titles, max_selections=st.session_state['num_experimental_max_docs'], key='multiselect_docs')
                st.session_state['key_mselect_qdocs'] = QDOCS
                if st.session_state['chk_preprocessed_topic']:
                    all_docs_existing_topics = []
                    for iq, qdoc in enumerate(QDOCS):
                        topic_df = get_topic_df(qdoc)
                        if topic_df is not None and len(topic_df) > 0:
                            existing_topics = eval(topic_df['combined_topics'].iloc[0])
                            # annot_existing_labels = [(f'doc{iq}_t{str(i)}', t.upper()) for i, t in enumerate(existing_topics)]
                            annot_existing_labels = [(f'D{iq}', t.upper()) for i, t in enumerate(existing_topics)]
                            annotated_text(annot_existing_labels)
                            all_docs_existing_topics.append({'topics': existing_topics})

                # Topic Modeling for Document Analyzer
                if 'topic_selection_label' not in st.session_state:
                    st.session_state['topic_selection_label'] = "Select Topics:"
                if 'candidate_labels' not in st.session_state:
                    st.session_state['candidate_labels'] = CANDIDATE_LABELS
                if 'merge_topics_done' not in st.session_state:
                    st.session_state['merge_topics_done'] = False
                if st.session_state['chk_preprocessed_topic'] and len(QDOCS) > 1:
                    if st.session_state['merge_topics_done']:
                        st.caption(f"""__NOTE:__ There are ({len(st.session_state['candidate_labels'])}) Focused Topics found. Your *"Output Options"* has been resetted back to defaults.""")
                    else:
                        st.caption(f"""Each document already has recommended topics, but the topic selection below uses system defaults. Click _"Merge All Documents’ Topics"_ to update and merge topics.""")
                    btn_update_topics = st.button("Merge All Documents' Topics") if not st.session_state['merge_topics_done'] else None
                    if btn_update_topics:
                        st.session_state['topic_selection_label'] = "Select Focused Topics:"
                        _, combined_topics = combine_topics(all_docs_existing_topics, llm)
                        topic_selection_list = [item['topic'] for item in combined_topics]
                        st.session_state['candidate_labels'] = topic_selection_list
                        st.session_state['merge_topics_done'] = True
                        st.toast(f'Focused Topics({len(topic_selection_list)}) merged successfully!', icon='🎉')
                        st.rerun()


                blank(lines=3)
                ###### 'Ask a Question:' feature is disabled for now, probably will be removed in the future as it is not necessary and redundant with the 'RAG ChatBot'
                QA = st.text_area("Ask a Question:", placeholder="Type your question here...", height=100, max_chars=5000, value=qa_val)
                st.session_state['key_txt_qa'] = QA
                _, center, _ = st.columns([5, 1, 5])
                center.subheader('OR')
                QT = st.multiselect(st.session_state['topic_selection_label'], st.session_state['candidate_labels'])
                st.session_state['key_mselect_qt'] = QT
            with cf2:
                blank()
                st.write("###### Output Options:")

                blank()
                if st.session_state['num_experimental_max_docs'] <= MAX_DOCS:
                    if 'chk_preprocessed_topic' in st.session_state and st.session_state['chk_preprocessed_topic']:
                        if 'chk_preview_annotated_docs' not in st.session_state:
                            st.session_state['chk_preview_annotated_docs'] = False
                        st.checkbox('Preview Annotated Documents', value=st.session_state['chk_preview_annotated_docs'], key='chk_preview_annotated_docs')

                    if 'chk_show_summary' not in st.session_state:
                        st.session_state['chk_show_summary'] = False
                    st.checkbox('Show Summary', value=(st.session_state['chk_show_summary'] or False), key=f'chk_show_summary')
                    blank()

                # st.write("Select Translation Language(s) you want to translate the summary to:")
                # for lang in SUPPORTED_LANGUAGES_T:
                #     st.checkbox(lang, value=False, key=f'chk_{lang.lower()}')
                doc_lang = st.selectbox("Select Translation Language Option:", SUPPORTED_LANGUAGES_T, index=(0 if newsroom == NEWSROOM_HS else 1))
                blank()
                with st.expander("Advanced Semantic Analysis:", expanded=False):
                    K = st.number_input('Number of Results(k) per Document:', min_value=MIN_RESULTS_PER_DOC, max_value=MAX_RESULTS_PER_DOC, value=K, step=5, key='number_input_k')
                    st.checkbox('Extract results(k) separately for each document', value=False, key='separate_documents_in_semantic_search')
                    if st.session_state['num_experimental_max_docs'] <= MAX_DOCS:
                        st.checkbox('Expand queries', value=False, key='expand_queries')

                if st.session_state['num_experimental_max_docs'] <= MAX_DOCS:
                    blank()
                    with st.expander("Advanced Text Analysis:", expanded=False):
                        st.checkbox('Show Wordclouds', value=False, key='chk_wordcloud')
                        st.checkbox('Show Sentiment Analysis', value=False, key='chk_sentiment')
                        st.checkbox('Extract Keywords', value=False, key='chk_keywords')
                        _, c_extract = st.columns([1, 15])
                        c_extract.number_input('Top Keywords:', min_value=MIN_NUM_KWORDS, max_value=MAX_NUM_KWORDS, value=10, step=5, key='top_keywords')
                        st.checkbox('Show Name Entity Recognition (NER)', value=False, key='chk_ner')

            if len(QT) > 0:
                Q = ', '.join(QT)
                QA = ''
            else:
                Q = QA

            cf1.markdown('___', unsafe_allow_html=True)
            if 'doc_analyzer_docanalysis' in st.session_state:
                btn_ask = cf1.button("Analyze Documents", disabled=True, type='primary')
            else:
                btn_ask = cf1.button("Analyze Documents", type='primary')

            if btn_ask and Q.strip() != '':
                if len(QDOCS) < 1:
                    st.toast("Please select at least one document for analysis.", icon='❌')
                else:
                    if 'doc_analyzer_query' not in st.session_state:
                        st.session_state['doc_analyzer_query'] = Q

                    # Semantic Search Results
                    if 'expand_queries' in st.session_state and st.session_state['expand_queries']:
                        expanded_queries = expand_query(Q, newsroom, doc_lang, 4, llm, 1)
                        st.session_state['expanded_queries'] = expanded_queries
                        if 'separate_documents_in_semantic_search' in st.session_state and st.session_state['separate_documents_in_semantic_search']:
                            results = semantic_search_expanded(Q, expanded_queries, k=K, collection=collection, titles=QDOCS, separate_documents=True)
                        else:
                            results = semantic_search_expanded(Q, expanded_queries, k=K, collection=collection, titles=QDOCS, separate_documents=False)
                    elif 'separate_documents_in_semantic_search' in st.session_state and st.session_state['separate_documents_in_semantic_search']:
                        results = semantic_search_separated_documents(Q, k=K, collection=collection, titles=QDOCS)
                    else:
                        results = semantic_search(Q, k=K, collection=collection, titles=QDOCS)
                    if 'doc_analyzer_result' not in st.session_state:
                        st.session_state['doc_analyzer_result'] = results

                    # Inspect Results
                    data_dict = {
                        'ids': results['ids'][0],
                        'distances': results['distances'][0],
                        'documents': results['documents'][0],
                        'title': [eval(str(m))['title'] for m in results['metadatas'][0]],
                        'url': [eval(str(m))['url'] for m in results['metadatas'][0]],
                        'metadata': results['metadatas'][0]
                    }

                    results_df = pd.DataFrame(data_dict)
                    cols = st.columns(results_df['title'].nunique())
                    unique_titles = results_df['title'].unique()

                    texts = [] # for HS analysis using joined chunks
                    for i in range(len(cols)):
                        with cols[i]:
                            title = unique_titles[i]
                            tmp_df = results_df[results_df['title'] == title]
                            source = ''
                            text = ''

                            for x in range(tmp_df.shape[0]):
                                source = f"Source: {tmp_df['url'].iloc[x]}"
                                text += '... ' + tmp_df['documents'].iloc[x] + '...\n\n'
                            texts.append(text) # for HS analysis using joined chunks

                            if 'chk_preview_annotated_docs' in st.session_state and st.session_state['chk_preview_annotated_docs']:
                                topic_df = get_topic_df(title)
                                if topic_df is not None and len(topic_df) > 0:
                                    doc_annot_texts = eval(topic_df['annotated_texts'].iloc[0])
                                    st.session_state[f'doc_analyzer_col{i}_annotations'] = doc_annot_texts
                                    # annotated_text(doc_annot_texts)

                            if 'chk_show_summary' in st.session_state and st.session_state['chk_show_summary']:
                                summary = ''
                                translation = ''
                                for il, lang in enumerate(SUPPORTED_LANGUAGES_T):
                                    if doc_lang == lang: #st.session_state[f'chk_{lang.lower()}']:
                                        if summary == '':
                                            summary = generate_summarization(text, llm)
                                            translation = summary
                                            if 'doc_analyzer_summary' not in st.session_state:
                                                st.session_state[f'doc_analyzer_col{i}_summary'] = summary
                                        if doc_lang != 'No translation' or doc_lang != 'English':
                                            translation = generate_translation(summary, lang, llm)

                                        if f'doc_analyzer_{lang}_translation' not in st.session_state:
                                            st.session_state[f'doc_analyzer_col{i}_{lang}_translation'] = translation
                                        break

                            if 'chk_sentiment' in st.session_state and st.session_state['chk_sentiment']:
                                sentiment_analysis = generate_sentiment_analysis(text, llm)
                                if 'doc_analyzer_sentiment_analysis' not in st.session_state:
                                    st.session_state[f'doc_analyzer_col{i}_sentiment_analysis'] = sentiment_analysis

                            if 'chk_keywords' in st.session_state and st.session_state['chk_keywords']:
                                top_k = st.session_state['top_keywords'] or 10
                                topic_labels = generate_keyword_labels(text, llm, top_k=top_k)
                                if 'doc_analyzer_topic_labels' not in st.session_state:
                                    st.session_state[f'doc_analyzer_col{i}_topic_labels'] = topic_labels

                    document_analysis = ''
                    if newsroom  == NEWSROOM_HS:
                        if 'num_experimental_max_docs' in st.session_state and st.session_state['num_experimental_max_docs'] > MAX_DOCS:
                            summaries = []
                            for text in texts:
                                summaries.append(generate_focused_summarization(Q, text, llm, newsroom, doc_lang))
                            texts = summaries
                        document_analysis = generate_document_analysis_hs(Q, unique_titles, texts, llm, advanced_prompt, doc_lang)
                    else:
                        document_analysis = generate_document_analysis(Q, results_df, llm, advanced_prompt)

                    # Translate the document analysis using the current user session's language of choice
                    if doc_lang != 'No translation' or doc_lang != 'English':
                        document_analysis = generate_translation(document_analysis, doc_lang, llm)

                    if 'doc_analyzer_docanalysis' not in st.session_state:
                        st.session_state['doc_analyzer_docanalysis'] = document_analysis

                    st.rerun()

            # Document Analyzer's feedback to be submitted and states has been updated
            if 'doc_analyzer_query' in st.session_state:
                Q = st.session_state['doc_analyzer_query']

            # Semantic Search Results
            if 'doc_analyzer_result' in st.session_state:
                results = st.session_state['doc_analyzer_result']

                # Inspect Results
                data_dict = {
                    'ids': results['ids'][0],
                    'distances': results['distances'][0],
                    'documents': results['documents'][0],
                    'title': [eval(str(m))['title'] for m in results['metadatas'][0]],
                    'url': [eval(str(m))['url'] for m in results['metadatas'][0]],
                    'metadata': results['metadatas'][0]
                }

                results_df = pd.DataFrame(data_dict)
                with st.expander("Semantic Data Analysis:", expanded=True):
                    st.subheader('Query:')
                    st.write(Q)
                    if 'expanded_queries' in st.session_state:
                        st.subheader('Expanded queries:')
                        queries_str = ''
                        for query in st.session_state['expanded_queries'][:-1]:
                            queries_str += f"{query}<br>"
                        st.write(f"""
                        {queries_str}
                        """, unsafe_allow_html=True)
                    st.subheader(f'Sources({results_df["title"].nunique()}):')
                    st.write('; '.join(results_df['title'].unique()))
                    st.subheader(f'Semantic Search Results Data (k={len(results_df)}):')
                    title_counts = results_df['title'].value_counts()
                    title_count_str = ''
                    for title, count in title_counts.items():
                        title_count_str += f"{title} (k={count})<br>"

                    st.write(f"""
                    {title_count_str}
                    """, unsafe_allow_html=True)
                    st.dataframe(results_df)
                    if 'chk_wordcloud' in st.session_state and st.session_state['chk_wordcloud']:
                        st.subheader('Word Clouds:')
                        st.pyplot(plot_wordcloud(results_df, 'documents'))

                cols = st.columns(results_df['title'].nunique())
                unique_titles = results_df['title'].unique()

                has_cols_idx = st.session_state['chk_show_summary'] or st.session_state['chk_sentiment'] or st.session_state['chk_keywords'] or st.session_state['chk_ner']
                for i in range(len(cols)):
                    with cols[i]:
                        title = unique_titles[i]
                        tmp_df = results_df[results_df['title'] == title]
                        source = ''
                        text = ''

                        for x in range(tmp_df.shape[0]):
                            source = f"Source: {tmp_df['url'].iloc[x]}"
                            text += '... ' + tmp_df['documents'].iloc[x] + '...\n\n'

                        if (newsroom != NEWSROOM_HS) and has_cols_idx:
                            st.header(title)
                            st.write(f"Document Result Index: {i}")
                            st.caption(f"Source: {results_df['url'].iloc[i]}")

                        if f'doc_analyzer_col{i}_annotations' in st.session_state:
                            doc_annot_texts = st.session_state[f'doc_analyzer_col{i}_annotations']
                            annotated_text(doc_annot_texts)

                        summary = ''
                        for il, lang in enumerate(SUPPORTED_LANGUAGES_T):
                            if doc_lang == lang: #st.session_state[f'chk_{lang.lower()}']:
                                if f'doc_analyzer_col{i}_{lang}_translation' in st.session_state:
                                    st.write('___')
                                    st.subheader(f'Summary:')
                                    translation = st.session_state[f'doc_analyzer_col{i}_{lang}_translation']
                                    st.write(translation)
                                break

                        if 'chk_sentiment' in st.session_state and st.session_state['chk_sentiment']:
                            st.subheader('Sentiment Analysis:')
                            if f'doc_analyzer_col{i}_sentiment_analysis' in st.session_state:
                                sentiment_analysis = st.session_state[f'doc_analyzer_col{i}_sentiment_analysis']
                                st.write(sentiment_analysis)
                                st.write('___')

                        if 'chk_keywords' in st.session_state and st.session_state['chk_keywords']:
                            st.subheader('Keywords:')
                            top_k = st.session_state['top_keywords'] or MAX_KEYWORDS
                            if f'doc_analyzer_col{i}_topic_labels' in st.session_state:
                                topic_labels = st.session_state[f'doc_analyzer_col{i}_topic_labels']
                                st.write(topic_labels)
                                st.write('___')

                        if 'chk_ner' in st.session_state and st.session_state['chk_ner']:
                            st.subheader('Name Entity Recognition *(NER)*:')
                            doc = SPACY_MODEL(text)
                            spacy_streamlit.visualize_ner(
                                doc,
                                labels = ENTITY_LABELS,
                                show_table = False,
                                title = '',
                                key=f'ner{i}'
                            )

                if 'doc_analyzer_docanalysis' in st.session_state:
                    document_analysis = st.session_state['doc_analyzer_docanalysis']
                    st.write('___')
                    st.write('# AIDOL Document Analysis:')
                    blank(lines=2)
                    st.markdown(document_analysis.replace('```markdown', '').replace('```', ''))

                    blank(lines=3)
                    st.caption("Was this helpful?")
                    set_docanalyzer_feedback(newsroom=newsroom, Q=Q, prompt=advanced_prompt, document_analysis=document_analysis, QDOCS=QDOCS)
                    st.write('___')

                    blank(lines=2)
                    _, c_reset, _ = st.columns([3, 3, 2])
                    c_reset.button("Reset Document Analyzer", on_click=reset_document_analyzer, type='primary')
                    if "docanalyzer_feedback_sent" in st.session_state and st.session_state.docanalyzer_feedback_sent:
                        st.toast('Document Analyzer feedback submitted successfully!', icon='🎉')
                        st.session_state.docanalyzer_feedback_sent = False

        # RAG ChatBot
        with tab2:
            QDOCS = []
            cf1, _, cf2 = st.columns([20, 1, 5])
            with cf1:
                QDOCS = st.multiselect("Select Documents:", df['title'].unique(), max_selections=st.session_state['num_experimental_max_docs'])

            with cf2:
                blank()
                st.write("###### Chat Options:")
                blank()
                lang = st.selectbox("Language Option:", SUPPORTED_LANGUAGES_T, index=0)

                blank()
                K = st.number_input('Number of Results(k) per Document:', min_value=MIN_RESULTS_PER_DOC, max_value=MAX_RESULTS_PER_DOC, value=DEFAULT_NUM_INPUT, step=5)
                blank()

            if len(QDOCS) > 0:
                cf1.caption(f'Sources: {", ".join(QDOCS)}')

            if len(QDOCS) < 1:
                cf1.markdown("*Note:* Please select at least one document so you could start the chat.")

            else:
                if lang != 'No translation' or lang != "English":
                    # titles = list(map(lambda x: generate_translation(x, llm, "English", lang), orig_titles))
                    feedback_caption = generate_translation("Was this helpful?", lang, llm)
                    user_prompt = generate_translation("Type your questions here...", lang, llm)
                    load_response = generate_translation("Loading a response...", lang, llm)
                else:
                    # titles = orig_titles
                    feedback_caption = 'Was this helpful?'
                    user_prompt = "Type your questions here..."
                    load_response = "Loading a response..."

                # Initialize title
                if "titles" not in st.session_state:
                    st.session_state['titles'] = None

                if "feedback" not in st.session_state:
                    st.session_state['feedback'] = None

                # Initiliaze spoken
                if "spoken" not in st.session_state:
                    st.session_state['spoken'] = False

                # Initialize total number of responses
                if "total_responses" not in st.session_state:
                    st.session_state['total_responses'] = 0

                with cf1:
                    blank(lines=2)
                    st.write("###### Chat:")

                    # Initialize chat history or reset history if you change documents
                    if "messages" not in st.session_state or st.session_state['titles'] != QDOCS:
                        try:
                            if len(st.session_state.messages) > 0:
                                st.session_state.total_responses += len(st.session_state.messages)
                        except:
                            pass
                        st.session_state.messages = []
                        st.session_state['titles'] = QDOCS
                        st.session_state['feedback'] = None

                    # Display chat messages from history
                    for i, message in enumerate(st.session_state.messages):
                        with st.chat_message(message['role']):
                            st.markdown(message['content'])
                            if message['role'] == "assistant":
                                titles = st.session_state['titles']
                                documents = ', '.join(titles)
                                question = st.session_state.messages[i-1]['content']
                                current_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                idx = st.session_state.total_responses
                                newsroom = st.session_state.newsroom["newsroom"]

                                st.caption(feedback_caption)
                                fbk_from = 'rag_chatbot_fbk'
                                streamlit_feedback(
                                    feedback_type="faces",
                                    key = f'comment_{i+idx}_{len(message)}_{"-".join(titles)}',
                                    optional_text_label="Please provide some more information here...",
                                    # max_text_length=1500,
                                    align='flex-start',
                                    kwargs={"fbk_from":fbk_from, "type": "rag_chatbot", "newsroom": newsroom, "question": question, "prompt": f"[same as the chat question]\n\n{question}", "llm_response": message['content'], "documents": documents, "feedback_time": current_date},
                                    on_submit=submit_feedback
                                )

                    # Accept user input
                    if prompt := st.chat_input(user_prompt):

                        # Translate the user prompt to english
                        if lang != 'No translation' or lang != "English":
                            prompt = generate_translation(prompt, lang, llm)

                        # Add user message to chat history
                        st.session_state.messages.append({"role": "user", "content": prompt})

                        # Display user message
                        with st.chat_message("user"):
                            st.markdown(prompt)

                        # Display response
                        with st.chat_message("assistant"):
                            with st.spinner(load_response):
                                # Semantic Search Results
                                response = ask_query(prompt, QDOCS, llm, collection, k=15)

                                # Translate the response if the language of choice is not in English
                                if lang != 'No translation' or lang != "English":
                                    response = generate_translation(response, lang, llm)

                                st.session_state.messages.append({"role": "assistant", "content": response})
                                st.markdown(response)

                        st.rerun()
                    blank(lines=2)
                    st.write('___')
                    if len(st.session_state.messages) > 0:
                        _, c_reset_chat, _ = st.columns([3, 3, 2])
                        c_reset_chat.button("Reset Chat History", on_click=reset_ragchatbot)

                    if 'ragchat_feedback_sent' in st.session_state and st.session_state.ragchat_feedback_sent:
                        st.toast('RAG Chat feedback submitted successfully!', icon='🎉')
                        st.session_state.ragchat_feedback_sent = False

