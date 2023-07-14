import whisper
import gradio as gr
import time
import warnings
import json
import openai
import os
import constants
from pathlib import Path
import torch
import re

"""# Defining Variables"""
os.environ["OPENAI_API_KEY"] = constants.APIKEY
openai.api_key = constants.APIKEY
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

def create_textfile(video_name, sr, timelag):
    result_text = model.transcribe(video_name, verbose=False, language="ja")
    text = result_text["text"]
    return [text] 

output_1 = gr.Textbox(label="Speech to Text")

gr.Interface(
    title = 'OpenAI Whisper and ChatGPT ASR Gradio Web UI',
    fn=create_textfile,
    inputs=[
        gr.inputs.Audio(source="microphone", type="filepath"),
    ],

    outputs=[
        output_1,#  output_2
    ],
    live=True).launch()

