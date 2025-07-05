import pandas as pd
import numpy as np
import nltk

#1.Coleta dos dados

colunas = ['ID', 'label', 'texto', 'assunto', 'Orador', 'Profissao', 'Estado', 'partido', 'n_majoritariamente_falsa', 'n_falsa', 'n_meio_falso_verdade', 'n_majoritariamente_verdade', 'n_mentira_absurda', 'contexto']

dataTreino = pd.read_csv('datasets/train.tsv', header=None, sep='\t', names=colunas)
dataTeste = pd.read_csv('datasets/test.tsv', header=None, sep='\t', names=colunas)
dataValid = pd.read_csv('datasets/valid.tsv', header=None, sep='\t', names=colunas)

#2.Preparação dos dados

#print(dataTreino.info())
#print(len(dataTreino))
dataTreino[['Profissao', 'Estado', 'contexto']] = dataTreino[['Profissao', 'Estado', 'contexto']].fillna('desconhecido')
dataTeste[['Profissao', 'Estado', 'contexto']] = dataTeste[['Profissao', 'Estado', 'contexto']].fillna('desconhecido')
dataValid[['Profissao', 'Estado', 'contexto']] = dataValid[['Profissao', 'Estado', 'contexto']].fillna('desconhecido')
dataTreino['texto_completo'] = (
    dataTreino['texto'] + ' ' + dataTreino['assunto'] + ' ' + dataTreino['contexto'] + ' ' + dataTreino['Profissao']
)
dataTeste['texto_completo'] = (
    dataTeste['texto'] + ' ' + dataTeste['assunto'] + ' ' + dataTeste['contexto'] + ' ' + dataTeste['Profissao']
)
dataValid['texto_completo'] = (
    dataValid['texto'] + ' ' + dataValid['assunto'] + ' ' + dataValid['contexto'] + ' ' + dataValid['Profissao']
)
#print(dataTreino.isnull().sum())
#print(dataTeste.isnull().sum())
#print(dataValid.isnull().sum())
#print(dataTreino['label'].value_counts())

dataTreino['texto_completo'] = dataTreino['texto_completo'].str.lower()
dataTeste['texto_completo'] = dataTreino['texto_completo'].str.lower()
dataValid['texto_completo'] = dataTreino['texto_completo'].str.lower()

mapa_labels = {'pants-fire':0, 'false':1, 'barely-true':2, 'half-true':3, 'mostly-true':4, 'true':5}
dataTreino['label'] = dataTreino['label'].map(mapa_labels)
dataTeste['label'] = dataTeste['label'].map(mapa_labels)
dataValid['label'] = dataValid['label'].map(mapa_labels)

