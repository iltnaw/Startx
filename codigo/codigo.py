import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from collections import Counter
import random # Mantido, mas np.random é o principal para DP

# Ignora warnings de versões futuras do Seaborn e outras bibliotecas
warnings.filterwarnings("ignore", category=FutureWarning)

# --- 1. Funções Auxiliares ---

def get_decade(date_str):
    """
    Extrai a década de uma string de data.
    Ex: '25/06/1990' -> 1990
    """
    if pd.isna(date_str) or not isinstance(date_str, str):
        return None
    try:
        # Extrai o ano e calcula a década
        year = int(date_str.split('/')[-1])
        return (year // 10) * 10
    except (ValueError, IndexError):
        return None

def generate_decade_histogram(df):
    """
    Gera um histograma de frequências de nascimento por década.
    """
    print("   - Gerando histograma de frequências por década...")
    df['decada'] = df['data nascimento'].apply(get_decade)

    decades = df.dropna(subset=['decada'])['decada'].astype(int)
    
    # Adição de verificação de dataframe vazio para evitar erros
    if decades.empty:
        print("   - Aviso: Não há dados válidos de década após a limpeza. Retornando histograma vazio.")
        return {}
    
    min_year = decades.min()
    max_year = decades.max()
    # Garante que o range de bins inclua a última década corretamente
    bins = range((min_year // 10) * 10, (max_year // 10) * 10 + 20, 10) 
    
    histogram_data = pd.cut(decades, bins=bins, right=False, include_lowest=True,
                            labels=[f'{d}-{d+9}' for d in bins[:-1]]).value_counts().sort_index()

    full_range_labels = [f'{d}-{d+9}' for d in range((min_year // 10) * 10, (max_year // 10) * 10 + 10, 10)]
    
    complete_histogram = pd.Series(0, index=pd.CategoricalIndex(full_range_labels, ordered=True, categories=full_range_labels))
    complete_histogram.update(histogram_data)

    print(f"   - Histograma real das décadas de nascimento:\n{complete_histogram.to_string()}")
    return complete_histogram.to_dict()

def plot_and_save_histogram(data, title, xlabel, ylabel, filename):
    """
    Plota e salva um gráfico de barras.
    """
    # Adição de verificação para histogramas vazios
    if not data:
        print(f"   - Aviso: Não há dados para plotar o histograma '{title}'. Pulando salvamento.")
        return
        
    plt.figure(figsize=(12, 7))
    keys = list(data.keys())
    values = list(data.values())
    
    sns.barplot(x=keys, y=values, palette='viridis', hue=keys, legend=False)
    
    for i, v in enumerate(values):
        # Arredonda para 2 casas decimais, mas exibe como inteiro se for quase um inteiro
        plt.text(i, v + 0.5, str(round(v, 2) if abs(v - round(v)) > 0.01 else int(round(v))), ha='center', va='bottom', fontsize=10)
            
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"   - Histograma salvo em: {filename}")


# --- 2. Implementação dos Mecanismos de Privacidade Diferencial ---

def laplace_mechanism(query_result, sensitivity, epsilon):
    """
    Aplica o mecanismo de Laplace para adicionar ruído a um resultado numérico.
    """
    # Parâmetro de escala da distribuição de Laplace
    b = sensitivity / epsilon
    # Gera ruído aleatório de Laplace
    laplace_noise = np.random.laplace(loc=0, scale=b)
    # Adiciona o ruído ao resultado
    noisy_result = query_result + laplace_noise
    return max(0, noisy_result)

def exponential_mechanism(score_function, candidates, epsilon, sensitivity):
    """
    Aplica o mecanismo exponencial para selecionar um item de um conjunto de forma privada.
    """
    # Calcula a pontuação para cada candidato
    scores = {c: score_function(c) for c in candidates}
    
    # Calcula as probabilidades de seleção para cada candidato
    max_exp_arg = 709 # Aprox. np.log(np.finfo(np.float64).max)
    
    probabilities = {}
    for c, s in scores.items():
        exp_arg = (epsilon * s) / (2 * sensitivity)
        exp_arg = min(exp_arg, max_exp_arg) 
        probabilities[c] = np.exp(exp_arg)
    
    # Normaliza as probabilidades para que somem 1
    total_prob = sum(probabilities.values())
    
    if total_prob == 0:
        normalized_probabilities = [1 / len(candidates)] * len(candidates) if len(candidates) > 0 else []
    else:
        normalized_probabilities = [p / total_prob for p in probabilities.values()]
    
    if len(candidates) > 0 and normalized_probabilities:
        chosen_candidate = np.random.choice(
            list(probabilities.keys()),
            p=normalized_probabilities
        )
    else:
        chosen_candidate = None
    
    return chosen_candidate, scores


# --- 3. Função Principal ---

def main():
    """
    Função principal que orquestra todo o processo de privacidade diferencial.
    """
    # *** ALTERAÇÃO CHAVE AQUI PARA GARANTIR RUÍDO DIFERENTE PARA CADA EPSILON ***
    # Definindo uma semente para reprodutibilidade da execução COMPLETA do script.
    # Se você RODAR O SCRIPT VÁRIAS VEZES, ele SEMPRE produzirá os MESMOS GRÁFICOS.
    # No entanto, DENTRO DE UMA ÚNICA EXECUÇÃO, o ruído para cada epsilon será DIFERENTE.
    np.random.seed(42) # A semente pode ser qualquer número inteiro.
    # Se você quiser que CADA EXECUÇÃO do script produza gráficos DIFERENTES,
    # remova a linha acima ou use np.random.seed(int(datetime.now().timestamp()))


    file_path = 'dados covid-ce 02.csv'
    output_dir = 'dp_results_2025_1'
    os.makedirs(output_dir, exist_ok=True)

    try:
        df = pd.read_csv(file_path, delimiter=';', low_memory=False, encoding='latin1') 
    except FileNotFoundError:
        print(f"Erro: Arquivo '{file_path}' não encontrado. Certifique-se de que está no mesmo diretório do script.")
        print("Você pode usar 'os.getcwd()' no Python para verificar o diretório de trabalho atual,")
        print("ou passar o caminho absoluto completo do arquivo.")
        return
    except Exception as e:
        print(f"Erro ao ler ou processar o arquivo CSV: {e}")
        return

    # Renomeia e padroniza colunas para consistência
    df.columns = df.columns.str.strip().str.lower().str.replace('_', ' ')
    df = df.rename(columns={
        'data nascimento': 'data nascimento',
        'raca cor': 'raca cor',
        'data_nascimento': 'data nascimento', 
        'raca_cor': 'raca cor',              
        'localidade': 'localidade'            
    })

    epsilon_values = [0.01, 0.1, 0.5, 1.0]

    # --- Seção 2.1: Mecanismo de Laplace ---
    print("\n" + "="*50)
    print("##### Seção 2.1: Mecanismo de Laplace #####")
    print("="*50 + "\n")
    
    real_histogram_decades = generate_decade_histogram(df)
    
    plot_and_save_histogram(
        real_histogram_decades,
        'Histograma Real - Frequência de Nascimentos por Década',
        'Década de Nascimento',
        'Frequência',
        os.path.join(output_dir, 'histograma_real_decadas.png')
    )

    sensitivity_laplace = 1
    print(f"\n1. Sensibilidade da consulta de histograma (contagem): {sensitivity_laplace}")

    print("\n2. Gerando versões anonimizadas com ruído de Laplace...")
    laplace_results_summary = {}
    laplace_df_results = pd.DataFrame(index=real_histogram_decades.keys())
    laplace_df_results['Real'] = list(real_histogram_decades.values())

    for epsilon in epsilon_values:
        noisy_histogram = {}
        for decade, count in real_histogram_decades.items():
            noisy_count = laplace_mechanism(count, sensitivity_laplace, epsilon)
            noisy_histogram[decade] = noisy_count
        
        laplace_results_summary[epsilon] = noisy_histogram
        laplace_df_results[f'ε={epsilon}'] = list(noisy_histogram.values())
        
        plot_and_save_histogram(
            noisy_histogram,
            f'Histograma Anonimizado - Laplace (ε={epsilon})',
            'Década de Nascimento',
            'Frequência com Ruído',
            os.path.join(output_dir, f'histograma_laplace_e{epsilon}.png')
        )
    
    print("\n--- Histograma real e anonimizados de Laplace gerados e salvos. ---\n")

    # --- Seção 2.2: Mecanismo Exponencial ---
    print("\n" + "="*50)
    print("##### Seção 2.2: Mecanismo Exponencial #####")
    print("="*50 + "\n")
    
    df['localidade_str'] = df['localidade'].astype(str)
    df_morrinhos = df[df['localidade_str'].str.contains('MORRINHOS', case=False, na=False)].copy()
    
    candidates = df['raca cor'].dropna().unique().tolist()
    
    real_race_counts = df_morrinhos['raca cor'].value_counts().to_dict()
    print(f"   - Frequências reais de Raça/Cor em MORRINHOS:\n{real_race_counts}")

    sensitivity_score = 1
    print(f"\n1. Sensibilidade da função de score (contagem): {sensitivity_score}")

    def score_function_morrinhos(race_color):
        return real_race_counts.get(race_color, 0)

    print("\n2 & 3. Aplicando o Mecanismo Exponencial 20 vezes para cada epsilon...")
    exponential_results = {}
    for epsilon in epsilon_values:
        print(f"\n   - Executando para ε = {epsilon}:")
        run_results = []
        
        for _ in range(20):
            if not candidates:
                chosen_race = "Nenhum candidato disponível"
            else:
                chosen_race, scores_calc = exponential_mechanism(
                    score_function_morrinhos,
                    candidates,
                    epsilon,
                    sensitivity_score
                )
            run_results.append(str(chosen_race)) 
            
        counts = Counter(run_results)
        exponential_results[epsilon] = counts
        print(f"     -> Resultados das 20 execuções:\n       {dict(counts)}")
        
    print("\n--- Resultados do Mecanismo Exponencial obtidos. ---")

    # --- Geração do Relatório de Resultados (otimizado) ---
    report_path = os.path.join(output_dir, 'relatorio_privacidade_diferencial.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("RELATÓRIO DE PRIVACIDADE DIFERENCIAL\n")
        f.write("======================================\n\n")
        f.write("### Mecanismo de Laplace (Histogramas de Décadas) ###\n")
        f.write(f"Sensibilidade da consulta de contagem: {sensitivity_laplace}\n\n")
        
        f.write("--- Frequências de Nascimentos por Década (Real vs. Anonimizada) ---\n")
        f.write(laplace_df_results.to_string())
        f.write("\n\n")
        
        f.write("### Mecanismo Exponencial (Raça/Cor em MORRINHOS) ###\n")
        f.write(f"Sensibilidade da função de score: {sensitivity_score}\n")
        f.write("Candidatos para a seleção: " + str(candidates) + "\n")
        f.write(f"Frequências reais de Raça/Cor em MORRINHOS: {real_race_counts}\n\n")
        
        f.write("--- Resultados da Consulta 'Qual é a raça/cor predominante?' (20 execuções) ---\n")
        for epsilon, counts in exponential_results.items():
            f.write(f"   - Epsilon = {epsilon}:\n")
            for race, count in counts.items():
                f.write(f"     - '{race}': {count} vezes\n")
            f.write("\n")
            
        f.write("\n======================================\n")
        f.write("Os histogramas e gráficos foram salvos no diretório 'dp_results_2025_1'.\n")
        
    print(f"\n--- Relatório de resultados salvo em: {report_path} ---")
    print("\n--- Processo de anonimização concluído com sucesso! ---")


if __name__ == "__main__":
    main()