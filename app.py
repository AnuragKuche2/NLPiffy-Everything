# importing streamlit
import streamlit as st

# importing NLP packages
import spacy
from textblob import TextBlob
from gensim.summarization import summarize

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer


# sumy helper function
def sumy_summarizer(docs):
    parser = PlaintextParser.from_string(docs, Tokenizer("english"))
    lex_summarizer = LexRankSummarizer()
    summary = lex_summarizer(parser.document, 3)
    summary_list = [str(sentence) for sentence in summary]
    result = " ".join(summary_list)
    return result

# text analyzer helper function
def text_analyzer(text):
    nlp = spacy.load('en')
    docs = nlp(text)
    # for token in docs:
    #     print(token.text)
    # tokens = [token.text for token in docs]
    token_op = [('"Tokens":{},\n "Lemma":{}'.format(token.text, token.lemma_)) for token in docs]
    return token_op

# entity extractor helper function
def entity_extrator(text):
    nlp = spacy.load('en')
    docs = nlp(text)
    tokens = [token.text for token in docs]
    entity_op = [(entity.text, entity.label_) for entity in docs.ents]
    data_op = ['"Tokens":{}, \n"Entities":{}'.format(tokens, entity_op)]
    return data_op


# importing data analysis packages
def main():
    """ NLP application with StreamLit"""
    st.title("NLPiffy streamlit")
    st.header("Natural language processing on the Go")

    # tokenization
    if st.checkbox("Tokenize and lemmatize!"):
        st.header("Tokenized text is : ")
        message = st.text_area("Enter the text to be tokkenized and lemmatized", "Type here:")
        if st.button("Tokenize and Lemmatize"):
            tokenlized_results = text_analyzer(message)
            st.json(tokenlized_results)

    # named entity
    if st.checkbox("Entity extraction:"):
        st.header("Entities in the given text are:")
        message = st.text_area("Enter your text below:", "Type here:")
        if st.button("Extract Entities"):
            entity_results = entity_extrator(message)
            st.json(entity_results)

    # sentiment analysis
    if st.checkbox("Sentiment Analysis"):
        st.header("Text to be analyzed")
        message = st.text_area("Enter your text to be analyzed:", "Type here:")
        if st.button("Know the Sentiment"):
            blob = TextBlob(message)
            senti_results = blob.sentiment
            st.success(senti_results)

    # text summarization
    if st.checkbox("what to Summarize your Text?"):
        st.header("Text to be summarized")
        message = st.text_area("Enter the text to be summarized below:", "Type here:")
        summary_options = st.selectbox('Choice your Summarizer', ('Gensim', "Sumy"))
        if st.button("Know the summary"):
            if summary_options == "Gensim":
                st.text("using Gensim..:.:")
                summary_result = summarize(message)
            elif summary_options == "Sumy":
                st.text("using Sumy..:.:")
                summary_result = sumy_summarizer(message)

            else:
                st.warning("Using Default Sammarizer")
                st.text("Using Gensim")
                summary_result = summarize(message)

            st.success(summary_result)

    st.sidebar.subheader("About the App")
    st.sidebar.text("NLPiffy APP with StreamLit")
    st.sidebar.info("Learning Streamlit is easy")


if __name__ == "__main__":
    main()
