import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import Counter
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# -----------------------
# Funções Auxiliares
# -----------------------

def get_decade(date_str):
    try:
        year = int(date_str.split("/")[-1])
        return (year // 10) * 10
    except:
        return None

def generate_decade_histogram(df):
    decades = df["data nascimento"].apply(get_decade).dropna().astype(int)
    bins = list(range(1910, 2030, 10))
    labels = [f"{b}-{b+9}" for b in bins[:-1]]
    hist = pd.cut(decades, bins=bins, right=False, labels=labels).value_counts().sort_index()
    full_hist = {label: hist.get(label, 0) for label in labels}
    return full_hist

def plot_histogram(data, title, xlabel, ylabel, filename):
    plt.figure(figsize=(12,6))
    sns.barplot(x=list(data.keys()), y=list(data.values()), palette="viridis")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def laplace_mechanism(count, sensitivity, epsilon):
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale)
    return max(0, count + noise)

def exponential_mechanism(score_fn, candidates, epsilon, sensitivity):
    scores = {c: score_fn(c) for c in candidates}
    exp_scores = {c: np.exp((epsilon * s) / (2 * sensitivity)) for c, s in scores.items()}
    total = sum(exp_scores.values())
    probabilities = [exp_scores[c] / total for c in candidates]
    return np.random.choice(candidates, p=probabilities)

# -----------------------
# Função Principal
# -----------------------

def main():
    np.random.seed(42)

    file_path = "dados covid-ce 02.csv"
    output_dir = "dp_results_2025_1"
    os.makedirs(output_dir, exist_ok=True)

    try:
        df = pd.read_csv(file_path, sep=";", encoding="latin1")
    except Exception as e:
        print("Erro ao ler o CSV:", e)
        return

    df.columns = df.columns.str.lower().str.strip()

    epsilons = [0.01, 0.1, 0.5, 1.0]
    sensitivity = 1

    # -----------------------
    # Mecanismo de Laplace
    # -----------------------
    print("\n=== Mecanismo de Laplace ===")

    hist_real = generate_decade_histogram(df)

    plot_histogram(
        hist_real,
        "Histograma Real - Décadas",
        "Década de Nascimento",
        "Contagem",
        os.path.join(output_dir, "histograma_real_decadas.png")
    )

    for eps in epsilons:
        noisy_hist = {k: laplace_mechanism(v, sensitivity, eps) for k,v in hist_real.items()}
        plot_histogram(
            noisy_hist,
            f"Histograma Laplace (ε={eps})",
            "Década de Nascimento",
            "Contagem com Ruído",
            os.path.join(output_dir, f"histograma_laplace_eps_{eps}.png")
        )
        print(f"Histograma com Laplace ε={eps}: {noisy_hist}")

    # -----------------------
    # Mecanismo Exponencial
    # -----------------------
    print("\n=== Mecanismo Exponencial ===")

    df["localidade"] = df["localidade"].astype(str)
    morrinhos = df[df["localidade"].str.contains("MORRINHOS", case=False, na=False)]

    possible_races = ["PARDA", "AMARELA", "BRANCA", "PRETA", "INDÍGENA"]
    race_counts = morrinhos["raca cor"].value_counts().to_dict()
    print("Contagem real MORRINHOS:", race_counts)

    def score_fn(race):
        return race_counts.get(race, 0)

    fig, axes = plt.subplots(2,2, figsize=(14,10))
    axes = axes.flatten()

    for i, eps in enumerate(epsilons):
        np.random.seed(100 + i)
        results = [exponential_mechanism(score_fn, possible_races, eps, sensitivity) for _ in range(20)]
        counts = Counter(results)
        sns.countplot(x=results, ax=axes[i], palette="pastel")
        axes[i].set_title(f"Exponencial ε={eps}")
        axes[i].set_xlabel("Raça/Cor")
        axes[i].set_ylabel("Frequência")

    plt.suptitle("20 Execuções do Mecanismo Exponencial por ε")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mecanismo_exponencial_execucoes.png"))
    plt.close()

    print("\nProcesso concluído. Resultados salvos em", output_dir)

if __name__ == "__main__":
    main()
