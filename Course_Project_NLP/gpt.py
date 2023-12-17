import os
import gradio as gr
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from transformers import AutoTokenizer, AutoModel

class Chatbot():
    def __init__(self,path):
        self.path=path
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(path, trust_remote_code=True).half().cuda()
        self.model = self.model.eval()