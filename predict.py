from cog import BasePredictor, Input
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import torch

class Predictor(BasePredictor):
    def setup(self):
        self.tokenizer = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-Qwen2-7B-instruct", trust_remote_code=True)
        self.model = AutoModel.from_pretrained("Alibaba-NLP/gte-Qwen2-7B-instruct", trust_remote_code=True)

    def last_token_pool(self, last_hidden_states, attention_mask):
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            seq_len = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size), seq_len]

    def predict(
        self,
        query: str = Input(description="The query to encode."),
        task_description: str = Input(description="Instruction or task context", default="Given a web search query, retrieve relevant passages that answer the query")
    ) -> list:
        print('here')
        return {
            "log": "predict start",
            "query": query,
            "task": task_description
        }
        # prompt = f"Instruct: {task_description}\nQuery: {query}"
        # batch = self.tokenizer([prompt], return_tensors="pt", padding=True, truncation=True, max_length=8192)
        # outputs = self.model(**batch)
        # embeddings = self.last_token_pool(outputs.last_hidden_state, batch["attention_mask"])
        # embeddings = F.normalize(embeddings, p=2, dim=1)
        # return embeddings[0].tolist()
