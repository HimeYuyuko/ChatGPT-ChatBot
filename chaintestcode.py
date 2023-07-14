import os
import openai
import gradio as gr
import os
import torch
import whisper
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
import constants
import json
import re
from pathlib import Path
os.environ["OPENAI_API_KEY"] = constants.APIKEY
openai.api_key = constants.APIKEY


start_sequence = "\nAI:"
restart_sequence = "\nHuman: "
prompt="The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly."

loader = DirectoryLoader("data/")
index = VectorstoreIndexCreator().from_loaders([loader])
def load_whisper_model(model_name):
 model_dir = Path("whisper_models")
 model_path = model_dir / f"{model_name}.pt"

 if not model_path.exists():
    os.makedirs(model_dir, exist_ok=True)
    model = whisper.load_model(model_name)
    torch.save(model, model_path)
 else:
    model = torch.load(model_path)

 return model
model = load_whisper_model("large")

def run_chain(query):
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
    output=run_chain(inp)
    history.append((input,output))
    return history,history

def create_textfile(audio):
    result_text = model.transcribe(audio, verbose=False, language="ja")
    return result_text['text']

blocks = gr.Blocks()
with blocks:
    chatbot=gr.Chatbot()
    message=gr.Textbox(placeholder=prompt)
    audio_input = gr.inputs.Audio(source="microphone", type="filepath")
    state = gr.State()
    submit=gr.Button('Click')
    text_button = gr.Button("transcribe")
    submit.click(converstion_history,inputs=[message,state],outputs=[chatbot,state])
    text_button.click(create_textfile, inputs=[audio_input], outputs=[message])
    
blocks.launch(debug=True)


