
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import torch
from outlines.models.transformers import Transformers

def forense():
    model_id, model, tokenizer, device = get_gpt2_model_and_tokenizer()
    outlinesModel = Transformers(model, tokenizer)
    print("finish")

def get_gpt2_model_and_tokenizer():
    torch.manual_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "gpt2"

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, add_prefix_space=True)
    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                return_dict_in_generate=True,
                                                pad_token_id=tokenizer.eos_token_id).to(device)
                                                
    return model_id, model, tokenizer, device

if __name__ == "__main__":
    forense()