from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration
import os

class RAGModelHandler:
    def __init__(self):
        print("Loading model... This might take a while.")
        model_name = "facebook/rag-token-nq"
        cache_dir = os.path.expanduser(
            "~/.cache/huggingface/transformers"
        )  # Default cache directory
        # You may need to adjust this path based on your specific environment or Hugging Face configuration
        print("Loading model... This might take a while.")

        # Check if the model is already downloaded
        if not self.is_model_downloaded(cache_dir, model_name):
            print(f"Downloading model {model_name}...")
            # Downloading model
            self.tokenizer = RagTokenizer.from_pretrained(model_name)
            self.retriever = RagRetriever.from_pretrained(
                model_name, index_name="exact"
            )
            self.model = RagTokenForGeneration.from_pretrained(
                model_name, retriever=self.retriever
            )
            print("Model downloaded and loaded successfully.")
        else:
            print(f"Model {model_name} already downloaded. Loading from cache...")
            # Load model from cache
            self.tokenizer = RagTokenizer.from_pretrained(
                model_name, local_files_only=True
            )
            self.retriever = RagRetriever.from_pretrained(
                model_name, index_name="exact", local_files_only=True
            )
            self.model = RagTokenForGeneration.from_pretrained(
                model_name, retriever=self.retriever, local_files_only=True
            )
            print("Model loaded successfully from cache.")

    def generate_response(self, query: str, max_length: int = 30):
        print(f"Generating response for query: {query}")
        input_ids = self.tokenizer(query, return_tensors="pt").input_ids
        output = self.model.generate(input_ids, max_length=max_length)
        print(f"Generated response: {output[0]}")
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def is_model_downloaded(self, cache_dir, model_name):
        # Logic to check if the model files are in the cache directory
        # This might involve checking for specific files or patterns in the cache directory
        # For simplicity, this function just checks if the directory is non-empty
        return os.path.exists(cache_dir) and len(os.listdir(cache_dir)) > 0
