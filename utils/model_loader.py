import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AddedToken

def load_model(base_model_path, model_path):
    if model_path.endswith(".pt"):
        _model = torch.load(model_path, map_location="cpu", weights_only=False)
        
        new_state_dict = {}
        for key in _model["module"].keys():
            if key.startswith("model."):
                new_state_dict[key[len("model."):]] = _model["module"][key]

        model = AutoModelForCausalLM.from_pretrained(base_model_path)
        model.load_state_dict(new_state_dict)
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)

        return model, tokenizer
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        return model, tokenizer 

if __name__ == "__main__":
    model, tokenizer = load_model("Qwen/Qwen3-8B", "/data/verl/ckpts/reflection_v4_fullft_all/best.ckpt/checkpoint/mp_rank_00_model_states.pt")
    model.save_pretrained("/data/verl/ckpts/reflection_v4_fullft_all")
    tokenizer.save_pretrained("/data/verl/ckpts/reflection_v4_fullft_all")
    print(model)
    print(tokenizer)