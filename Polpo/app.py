import re
import time
import streamlit as st
from transformers import pipeline, Conversation, AutoTokenizer
from langdetect import detect
import torch

print(f"Is CUDA available: {torch.cuda.is_available()}")
# True
print(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
# Tesla T4

# choose your model here by setting model_chosen_id equal to 1 or 2
model_chosen_id = 2
model_name_options = {
    1: "meta-llama/Llama-2-13b-chat-hf",
    2: "BramVanroy/Llama-2-13b-chat-dutch"
}
model_chosen = model_name_options[model_chosen_id]

my_config = {'model_name': model_chosen, 'do_sample': True, 'temperature': 0.1, 'repetition_penalty': 1.1, 'max_new_tokens': 500, }
print(f"Selected model: {my_config['model_name']}")
print(f"Parameters are: {my_config}")

def count_words(text):
    # Use a simple regular expression to count words
    words = re.findall(r'\b\w+\b', text)
    return len(words)

def generate_with_llama_chat(my_config):    
    # get the parameters from the config dict
    do_sample = my_config.get('do_sample', True)
    temperature = my_config.get('temperature', 0.1)
    repetition_penalty = my_config.get('repetition_penalty', 1.1)
    max_new_tokens = my_config.get('max_new_tokens', 500)
    
    start_time = time.time()
    model = my_config['model_name']
    tokenizer = AutoTokenizer.from_pretrained(model)
    
    chatbot = pipeline("conversational",model=model, 
                       tokenizer=tokenizer,
                       do_sample=do_sample, 
                       temperature=temperature, 
                       repetition_penalty=repetition_penalty,
                       #max_length=2000,
                       max_new_tokens=max_new_tokens, 
                       #model_kwargs={"device_map": "auto","load_in_8bit": True}
                      )  #, "src_lang": "en", "tgt_lang": "nl"})  does not work!
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Loading the model: {elapsed_time} seconds")
    return chatbot
    
def get_answer(chatbot, input_text):
    start_time = time.time()
    print(f"Processing the input\n {input_text}\n")
    print('Processing the answer....')
    conversation = Conversation(input_text)
    print(f"Conversation(input_text): {conversation}")
    output = (chatbot(conversation))[1]['content']
    
    #Add the last print statement to the output variable
    output += f"\nAnswered in {elapsed_time:.1f} seconds, Nr generated words: {count_words(output)}"
    
    return output
    



chatbot = generate_with_llama_chat(my_config)
text = st.text_area("Enter text to summarize here.")

if text:
    out = get_answer(chatbot, text)
    st.json(out)