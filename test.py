import os
import openai
import gradio as gr

openai.api_key = "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"

start_sequence = "\nAI:"
restart_sequence = "\nHuman: "
prompt="The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly."
def openai_create(prompt):
    response = openai.Completion.create(
    model="text-davinci-003",
    prompt = prompt,
    temperature=0.9,
    max_tokens=150,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0.6,
    stop=[" Human:", " AI:"]
        )
    return response.choices[0].text 

def converstion_history(input,history):
    history = history or []
    s= list(sum(history,()))
    s.append(input)
    inp = ' '.join(s)
    output=openai_create(inp)
    history.append((input,output))
    return history,history

blocks = gr.Blocks()
with blocks:
    chatbot=gr.Chatbot()
    message=gr.Textbox(placeholder=prompt)
    state = gr.State()
    submit=gr.Button('Click')
    submit.click(converstion_history,inputs=[message,state],outputs=[chatbot,state])
    
blocks.launch(debug=True)