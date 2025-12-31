##### EXPLORANDO A BIBLIOTECA STATSMODEL 

1. Importe o dataset Titanic (v2) "titanic_v2.csv". 
#####
2. Obtenha informação relativa aos tipos de dados do dataset (método info()). 
#####
3. A partir do dataset original construa um dataset apenas com as colunas "Survived", "Age", "Pclass" e "Sex". 
#####
4. Remova do dataset anterior os valores NAN. 
#####
5. Defina o modelo "[Survived] ~ b0 + b1*[Age] + b2*[Pclass] + b3*[Sex] + e" usando um modelo linear do tipo OLS. 
#####
6. Ajuste as variáveis do modelo. 
#####
7. Imprima o resumo do modelo ajustado.
#####
8. Construa um modelo logístico usando como variável resposta "Survived" e variáveis preditoras "Age", "Pclass" e "Sex". 
#####
9. Separe o dataset em dois conjuntos, um de treino e outro de teste. 
#####

10. Efetue uma previsão no conjunto de teste.



Comentários:

1 - pd.read_csv('titanic_v2.csv'). Este comando depende sempre do ficheiro estar no mesmo diretório de execução (não é errado, mas convém garantir que efetivamente está).

5 - O modelo está corretamente definido: Survived ~ Age + Pclass + Sex. Em OLS, a interpretação de Sex como variável categórica está ok mas podia explicitar C(Sex).

8 - Parcialmente correto (6/10) - Neste ponto há um ligeiro desvio do que é pedido no enunciado porque utiliza LogisticRegression do scikit-learn e não statsmodels. O enunciado pede “explorar a biblioteca Statsmodels” e um “modelo logístico”, portanto o esperado seria algo como statsmodels (smf.logit / sm.Logit). A preparação de Sex para numérico (map) está correta para sklearn mas não segue o objetivo do enunciado. 
