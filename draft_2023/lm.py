from transformers import GPTNeoXTokenizerFast, GPTNeoXForCausalLM

from utils import lift, fsa_from_samples, estimate_entropy, get_samples

tokenizer = GPTNeoXTokenizerFast.from_pretrained('EleutherAI/pythia-19m')
model = GPTNeoXForCausalLM.from_pretrained('EleutherAI/pythia-19m')

def sample_from_lm(tokenizer, model, string, num, max_length=30):
    """Generate num samples from a causal LM"""
    inputs = tokenizer(string, return_tensors="pt")
    outputs = model.generate(**inputs, do_sample=True, max_length=max_length, num_return_sequences=num)
    # print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    return outputs

def n_gram(seq, n):
    """Generate n-gram states (e.g. n=2 gives bigrams) from a sequence sampled from an LM"""
    return [tuple(seq[max(0, i - n + 1):i + 1]) for i, y in enumerate(seq)]
    
def fsa_from_lm(tokenizer, model, string="I", num=20, gram=100):
    """Create an FSA from LM generations"""
    generations = sample_from_lm(tokenizer, model, string, num)
    mod = [tuple(n_gram(x, gram) if gram > 1 else x) for x in generations.tolist()]
    return fsa_from_samples(mod) 

def main():
    print(estimate_entropy(*fsa_from_lm(tokenizer, model)))

if __name__ == '__main__':
    main()