from transformers import GPTNeoXTokenizerFast, GPTNeoXForCausalLM

from utils import lift, fsa_from_samples, estimate_entropy, get_samples

tokenizer = GPTNeoXTokenizerFast.from_pretrained('EleutherAI/pythia-19m')
model = GPTNeoXForCausalLM.from_pretrained('EleutherAI/pythia-19m')

def sample_from_lm(tokenizer, model, string, num):
    """Generate num samples from a causal LM"""

    inputs = tokenizer(string, return_tensors="pt")
    outputs = model.generate(**inputs, do_sample=True, max_length=30, num_return_sequences=num)
    # print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    return outputs
    
def fsa_from_lm(tokenizer, model, string="I", num=10):
    generations = sample_from_lm(tokenizer, model, string, num)
    return fsa_from_samples([tuple(x) for x in generations.tolist()]) 

def main():
    print(estimate_entropy(*fsa_from_lm(tokenizer, model)))

if __name__ == '__main__':
    main()