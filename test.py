import random
import numpy
import math
import matplotlib.pyplot as plt

outcomes = 5
iterations = 10000

def generate_dist(n, total):
    dividers = sorted(numpy.random.uniform(0, total, size=(n - 1)))
    return [a - b for a, b in zip(dividers + [total], [0] + dividers)]

def estimate_entropies(prob, outcomes, sample, iterations):
    tot = [prob[0]] * outcomes
    for i in range(1, outcomes):
        tot[i] = tot[i - 1] + prob[i]

    print(prob)
    print(tot)

    print(f"Running {iterations} iterations, sampling {sample}.")

    true_entropy = -sum([x * math.log(x, 2) for x in prob])
    mle_entropy = 0
    horvitz_thompson_entropy = 0
    chao_shen_entropy = 0

    for _ in range(iterations):
        S = [0] * outcomes
        f_1 = 0
        for i in range(sample):
            res = random.random()
            for c, j in enumerate(tot):
                if j >= res:
                    S[c] += 1
                    if S[c] == 1: f_1 += 1
                    elif S[c] == 2: f_1 -= 1
                    break

        S = [x / sample for x in S]
        C = 1 - (f_1 / sample)

        mle_entropy -= sum([x * math.log(x, 2) if x else 0 for x in S])
        horvitz_thompson_entropy -= sum([x * math.log(x, 2) / (1 - (1 - x)**sample) if x else 0 for x in S])
        chao_shen_entropy -= sum([C * x * math.log(C * x, 2) / (1 - (1 - C * x)**sample) if x else 0 for x in S])

    mle_entropy /= iterations
    horvitz_thompson_entropy /= iterations
    chao_shen_entropy /= iterations

    # print(f"True entropy: {true_entropy}")
    # print(f"Average MLE entropy: {mle_entropy}")
    # print(f"Average H-T entropy: {horvitz_thompson_entropy}")
    # print(f"Average C-S entropy: {chao_shen_entropy}")
    return (true_entropy, mle_entropy, horvitz_thompson_entropy, chao_shen_entropy)
    

def main():
    outcomes = 20
    prob = generate_dist(outcomes, 1)
    true, mle, ht, cs = [], [], [], []
    x = list(range(outcomes + 1, 1000, 10))
    for sample in range(outcomes + 1, 1000, 10):
        a, b, c, d = estimate_entropies(prob, outcomes, sample, 1000)
        true.append(a)
        mle.append(b)
        ht.append(c)
        cs.append(d)
    plt.plot(x, true)
    plt.plot(x, mle)
    plt.plot(x, ht)
    plt.plot(x, cs)
    plt.legend(["True", "MLE", "Horvitz-Thompson", "Chao-Shen"])
    plt.xlabel("Sample size")
    plt.ylabel("(Estimated) Entropy (in bits)")
    plt.title(f"Entropy estimations for random probability distribution, {outcomes} outcomes")
    plt.show()
    

if __name__ == "__main__":
    main()