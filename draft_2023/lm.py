from transformers import GPTNeoXTokenizerFast, GPTNeoXForCausalLM, GPT2Tokenizer
import json
from tqdm import tqdm
import pandas as pd
import glob

from plotnine import ggplot, geom_line, geom_point, aes, stat_smooth, facet_wrap, theme, element_text
from plotnine.scales import scale_y_log10, scale_x_log10
from plotnine.guides import guide_legend, guides

from utils import lift, fsa_from_samples, estimate_entropy, get_samples

tokenizer = GPTNeoXTokenizerFast.from_pretrained('EleutherAI/pythia-19m')
model = GPTNeoXForCausalLM.from_pretrained('EleutherAI/pythia-19m')

def sample_from_lm(tokenizer: GPTNeoXTokenizerFast, model: GPTNeoXForCausalLM, string: str, num: int, max_length: int=30):
    """Generate num samples from a causal LM"""
    inputs = tokenizer(string, return_tensors="pt")
    print(inputs)
    outputs = model.generate(**inputs, do_sample=True, max_length=max_length, num_return_sequences=num)
    # print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    return outputs

def n_gram(seq: list[int], n: int):
    """Generate n-gram states (e.g. n=2 gives bigrams) from a sequence sampled from an LM"""
    res = [seq[max(0, i - n + 1):i + 1] for i, y in enumerate(seq)]
    return tuple([((x[-1],), tuple(x)) for x in res])
    
def fsa_from_lm(tokens: list[list[int]], gram: int=100, acyclic: bool=True):
    """Create an FSA from LM generations"""
    mod = [tuple([(x, x if acyclic else tuple(sent[:i])) for x in sent]) for i, sent in enumerate(tokens)]
    return fsa_from_samples(mod) 

def tokenize_from_file(file: str):
    """Read a jsonl file from the GPT-2 outputs dataset and tokenize it"""
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    with open(file, 'r') as fin:
        data = [json.loads(line) for line in fin]
    text = [row["text"] for row in data]
    tokens = tokenizer(text)["input_ids"]
    return tokens

def plot_convergence():
    res = []

    X = []
    for t in range(4): X.extend(list(range(2 * 10**t, min(6 * 10**3, 11 * 10**t), max(1, 10**t))))
    print(X[:100])

    for file in glob.glob('data/gpt2/*'):
        tokens = tokenize_from_file(file)
        print(file, "Tokenized")
        for num in tqdm(X):
            r = estimate_entropy(*fsa_from_lm(tokens[:num], gram=1), baseline=False)
            for estimator, val in r.items():
                print(f'{estimator:<30}: {val:>7.4f} nats')
                res.append({
                    'samples': num,
                    'method': estimator,
                    'entropy': val,
                    'dataset': file.split('/')[-1]
                })

    df = pd.DataFrame(res)
    plot = (ggplot(df, aes(x='samples', y='entropy', color='method',))
        + geom_line(stat='summary')
        + facet_wrap('~dataset', nrow=2, ncol=3)
        # + scale_y_log10()
        + scale_x_log10()
        + theme(legend_title=element_text(size=0, alpha=0),
            axis_text_x=element_text(rotation=45), legend_position=(0.8, 0.2))
        + guides(color=guide_legend(ncol=1)))
    plot.draw(show=True)
    plot.save(filename='plots/gpt.pdf', height=3, width=4)

def main():
    plot_convergence()

if __name__ == '__main__':
    main()
    # plot_convergence("data/gpt2/small")