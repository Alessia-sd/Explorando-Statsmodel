import pandas as pd
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def main():
    ### 1 Importe o dataset titanic_v2
    df = pd.read_csv('titanic_v2.csv')

    ### 2 Obtenha informação relativa aos tipos de dados
    print("--- Questão 2: Informações do Dataset ---")
    df.info()

    ### 3 Construa um dataset apenas com as colunas selecionadas
    df_resumo = df[['Survived', 'Age', 'Pclass', 'Sex']].copy()

    ### 4 Remova os valores NAN
    df_resumo.dropna(inplace=True)

    ### 5, 6 e 7 Definir, Ajustar e Imprimir o modelo OLS
    ### O statsmodels trata strings (Sex) como variáveis categóricas automaticamente
    modelo_ols = smf.ols(formula='Survived ~ Age + Pclass + Sex', data=df_resumo)
    resumo_ols = modelo_ols.fit()

    print("\n--- Questão 7: Resumo do Modelo Linear (OLS) ---")
    print(resumo_ols.summary())

    ### 8 Construa um modelo logístico
    ### Para o Scikit-Learn, precisamos converter 'Sex' em números (0 e 1)
    df_resumo['Sex_numeric'] = df_resumo['Sex'].map({'male': 0, 'female': 1})

    X = df_resumo[['Age', 'Pclass', 'Sex_numeric']]
    y = df_resumo['Survived']

    ### 9 Separe o dataset em dois conjuntos, um de treino e outro de teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    modelo_logistico = LogisticRegression()
    modelo_logistico.fit(X_train, y_train)

    ### 10 Efetue uma previsão no conjunto de teste <3
    previsoes = modelo_logistico.predict(X_test)

    print("\n--- Questão 10: Resultado das Previsões (Modelo Logístico) ---")
    print(classification_report(y_test, previsoes))

if __name__ == "__main__":
    main()