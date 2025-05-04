import pandas as pd

# Função para carregar os dados
def carregar_dados():
    # Carregamento dos dados
    avaliacoes = pd.read_csv('datasets/ratings_small.csv')
    metadados = pd.read_csv('datasets/movies_metadata.csv', low_memory=False)

    return avaliacoes, metadados

# Função para pré-processar os dados
def preprocessar_dados(avaliacoes, metadados):
    # Pré-processamento dos dados
    avaliacoes = avaliacoes[['userId', 'movieId', 'rating']]
    metadados = metadados[['id', 'title', 'genres']]

    metadados.loc[:, 'id'] = pd.to_numeric(metadados['id'], errors='coerce')
    metadados = metadados.rename(columns={'id': 'movieId'})
    metadados = metadados.dropna(subset=['movieId'])
    metadados['movieId'] = metadados['movieId'].astype(int)

    # Unificando avaliações e metadados
    dados_combinados = pd.merge(avaliacoes, metadados, on='movieId')
    dados_combinados = dados_combinados[dados_combinados['rating'] >= 3.0]

    return dados_combinados

# Função para gerar combinações de itemsets
def gerar_combos(itens, k):
    resultado = []
    def backtrack(inicio, atual):
        if len(atual) == k:
            resultado.append(atual[:])
            return
        for i in range(inicio, len(itens)):
            atual.append(itens[i])
            backtrack(i + 1, atual)
            atual.pop()
    backtrack(0, [])
    return resultado

# Funções para cálculo de suporte, confiança e lift
def calcular_suporte(itemset, dataset):
    return sum(1 for transacao in dataset if all(i in transacao for i in itemset)) / len(dataset)

def calcular_confianca(X, Y, dataset):
    cont_X = sum(1 for t in dataset if all(x in t for x in X))
    cont_XY = sum(1 for t in dataset if all(x in t for x in X + Y))
    return cont_XY / cont_X if cont_X > 0 else 0

def calcular_lift(X, Y, dataset):
    suporte_X = calcular_suporte(X, dataset)
    suporte_Y = calcular_suporte(Y, dataset)
    conf = calcular_confianca(X, Y, dataset)
    return conf / suporte_Y if suporte_Y > 0 else 0

# Função para encontrar os itemsets frequentes
def encontrar_itemsets(dataset, suporte_min):
    contagem = {}
    for trans in dataset:
        for item in trans:
            contagem[item] = contagem.get(item, 0) + 1
    total = len(dataset)
    nivel1 = [(tuple([item]), cont / total) for item, cont in contagem.items() if cont / total >= suporte_min]

    niveis = {}
    if not nivel1:
        return niveis
    niveis[1] = nivel1

    k = 2
    itemsets_atuais = [list(t[0]) for t in nivel1]
    while itemsets_atuais:
        itens = sorted(set(i for grupo in itemsets_atuais for i in grupo))
        candidatos = gerar_combos(itens, k)
        freq = []
        for c in candidatos:
            s = calcular_suporte(c, dataset)
            if s >= suporte_min:
                freq.append((tuple(c), s))
        if not freq:
            break
        niveis[k] = freq
        itemsets_atuais = [list(f[0]) for f in freq]
        k += 1
    return niveis

# Função para gerar as regras de associação
def gerar_regras(freq_itemsets, dataset, conf_min):
    regras = []
    for itemset, suporte in freq_itemsets:
        if len(itemset) < 2:
            continue
        itens = list(itemset)
        for i in range(1, len(itens)):
            for ant in gerar_combos(itens, i):
                cons = [i for i in itens if i not in ant]
                if not cons:
                    continue
                conf = calcular_confianca(ant, cons, dataset)
                if conf >= conf_min:
                    regras.append((tuple(ant), tuple(cons), round(conf, 2), round(calcular_lift(ant, cons, dataset), 2)))
    return regras

# Função para gerar recomendações
def gerar_recomendacao(usuarios_filmes, df_regras):
    usuario_exemplo = list(usuarios_filmes.keys())[0]  # Pega o primeiro usuário
    filmes_usuario = usuarios_filmes[usuario_exemplo]
    ultimo_filme = filmes_usuario[-1]  # Último filme curtido

    print(f"\nUsuário: {usuario_exemplo}")
    print(f"Último filme curtido: {ultimo_filme}")

    # Buscar regras onde o antecedente contém apenas o último filme assistido
    recomendacoes = df_regras[df_regras['Se tiver'] == (ultimo_filme,)]
    recomendacoes_ordenadas = recomendacoes.sort_values(by='Confiança', ascending=False)

    if not recomendacoes_ordenadas.empty:
        print("\nFilmes recomendados com base no último filme assistido:")
        for _, linha in recomendacoes_ordenadas.iterrows():
            print(f"Recomendado: {linha['Então sugerir']} (Confiança: {linha['Confiança']}, Lift: {linha['Lift']})")
    else:
        print("\nNenhuma recomendação encontrada para o último filme assistido.")

# main.py (parte principal)

# Carregar e pré-processar os dados
avaliacoes, metadados = carregar_dados()
dados_combinados = preprocessar_dados(avaliacoes, metadados)

# Mapeamento de filmes curtidos por usuário
usuarios_filmes = dados_combinados.groupby('userId')['title'].apply(list).to_dict()
lista_transacoes = list(usuarios_filmes.values())

# Parâmetros do algoritmo Apriori
suporte_minimo = 0.3
confianca_minima = 0.4

# Encontrar itemsets frequentes
itemsets_encontrados = encontrar_itemsets(lista_transacoes, suporte_minimo)

# Exibição dos itemsets frequentes
todos_itemsets = []
for nivel, itemsets in itemsets_encontrados.items():
    for itemset, suporte in itemsets:
        todos_itemsets.append({"Nível": nivel, "Itemset": itemset, "Suporte": round(suporte, 2)})
df_itemsets = pd.DataFrame(todos_itemsets)
print("Itens Frequentes Descobertos:")
print(df_itemsets)

# Gerar as regras de associação
todos_itemsets_flat = [i for sub in itemsets_encontrados.values() for i in sub]
regras = gerar_regras(todos_itemsets_flat, lista_transacoes, confianca_minima)

# Exibição das regras de associação
dados_regras = []
for ant, cons, conf, lift in regras:
    dados_regras.append({
        "Se tiver": ant,
        "Então sugerir": cons,
        "Confiança": conf,
        "Lift": lift
    })

df_regras = pd.DataFrame(dados_regras)
print("\nRegras de Assossiação:")
print(df_regras)

# Gerar recomendação para um usuário
gerar_recomendacao(usuarios_filmes, df_regras)
