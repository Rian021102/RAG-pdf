# CHAT WITH YOUR PDF 
## RAG TO CHAT WITH YOUR PDF
### PROJECT SCOPE
This project aims to give you a simple RAG application with two options:
1. OPEANAI Chatgpt
2. LLAMA3.2 via OLLAMA

#### 1. CHATGPT
Get your api key. We suggest you to store you OPENAI API KEY like this
##### ON MAC
echo "export OPENAI_API_KEY='yourkey'" >> ~/.zshrc
source ~/.zshrc
##### ON Windows
setx OPENAI_API_KEY "<yourkey>"

#### 2. OLLAMA
Install OLLAMA. Gd to this url: https://ollama.com/

download OLLAMA based on your OS
on terminal type: OLLAMA PULL LLAMA3.2 (if you want to use LLAMA3.2, go to the OLLAMA github to see all available models). 
And also, pull the embedding models. Go to the models and see the list of the embedding models provided by OLLAMA

### HOW TO USE THIS APPLICATIONS:
on your terminal type: 
git clone https://github.com/Rian021102/RAG-pdf
install all dependencies
pip install -r requirements.txt