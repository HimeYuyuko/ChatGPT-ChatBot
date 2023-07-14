import os
import openai
import gradio as gr
import os
import sys
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
import langid
import constants
os.environ["OPENAI_API_KEY"] = constants.APIKEY
openai.api_key = constants.APIKEY


start_sequence = "\nAI:"
restart_sequence = "\nHuman: "
prompt="The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly."

loader = DirectoryLoader("data/")
index = VectorstoreIndexCreator().from_loaders([loader])

def detect_language(text):
    lang = langid.classify(text)
    return lang

def run_chain(query,lang):
    chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-3.5-turbo"),
        retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
    )
    
    return chain.run(query)

def converstion_history(input,history):
    history = history or [] 
    s= list(sum(history,()))
    s.append(input)
    inp = ' '.join(s)
    lang=detect_language(inp)
    output=run_chain(inp,lang)
    history.append((input,output))
    return history,history

blocks = gr.Blocks()
with blocks:
    chatbot=gr.Chatbot()
    message=gr.Textbox(placeholder=prompt)
    state = gr.State()
    submit=gr.Button('Click')
    audio_input = gr.inputs.Audio(source="microphone", type="filepath")
    submit.click(converstion_history,inputs=[message,state],outputs=[chatbot,state])
    
blocks.launch(debug=True)