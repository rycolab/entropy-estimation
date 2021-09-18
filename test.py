import random
import numpy
import math
import matplotlib.pyplot as plt
import entropy

outcomes = 20
iterations = 100
max_sample = 1000

def generate_dist(n, total):
    dividers = sorted(numpy.random.uniform(0, total, size=(n - 1)))
    return [a - b for a, b in zip(dividers + [total], [0] + dividers)]

def estimate_entropies(prob, outcomes, N, iterations):
    tot = [prob[0]] * outcomes
    for i in range(1, outcomes):
        tot[i] = tot[i - 1] + prob[i]

    print(prob)
    print(tot)

    print(f"Running {iterations} iterations, sampling {N}.")

    funcs = [entropy.mle, entropy.horvitz_thompson, entropy.chao_shen,
        entropy.miller_madow, entropy.jackknife]
    entropies = [0] * len(funcs)
    true_entropy = -sum([x * math.log(x, 2) for x in prob])

    for _ in range(iterations):
        sample = []
        for i in range(N):
            res = random.random()
            for c, j in enumerate(tot):
                if j >= res:
                    sample.append(c)
                    break
        for i, estimator in enumerate(funcs):
            entropies[i] += estimator(sample)

    entropies = [x / iterations for x in entropies]

    return [true_entropy] + entropies
    

def main():
    prob = generate_dist(outcomes, 1)
    estimators = ["True", "MLE", "Horvitz-Thompson", "Chao-Shen", "Miller-Madow", "Jackknife"]

    y = []
    x = list(range(outcomes + 1, max_sample, 10))
    for sample in range(outcomes + 1, max_sample, 10):
        y.append(estimate_entropies(prob, outcomes, sample, iterations))
    for i in range(len(estimators)):
        plt.plot(x, [z[i] for z in y])

    plt.legend(estimators)
    plt.xlabel("Sample size")
    plt.ylabel("(Estimated) Entropy (in bits)")
    plt.title(f"Entropy estimations for random probability distribution, {outcomes} outcomes")
    plt.show()
    

if __name__ == "__main__":
    main()