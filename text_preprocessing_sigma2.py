import json
import os
import re
from bs4 import BeautifulSoup

def clean_ementa(texto):
    # Conferi 250 ementas. Parece estar funcionando perfeitamente. Houve erro em uma unica ementa que havia um cabecalho.
    texto = BeautifulSoup(texto, "lxml").text.replace('\xa0',' ')
    texto = texto.replace('E M E N T A', ' ')
    #Algumas ementas possuem acórdão depois
    texto = re.split('\n *ACÓRDÃO\n:? *', texto)[0]
    texto = re.sub('^EMENTA', ' ', texto.strip())
    texto = re.sub('(\n ?)+', '\n', texto.strip())
    return texto.strip()
def remover_citação_de_julgado(texto):
    # Ainda falta testar bastante este algoritmo
    texto = re.sub('[\-ºA-Z\.,À-ÿ0-9 \\\/]{50,300}.{20,3000}\(.{20,100}[0-9\.]{6,20}.{20,100}\)', ' ', texto)
    return texto
def clean_voto(texto):
    # Remover cabeçalho e outras coisas antes do voto
    texto = BeautifulSoup(texto, "lxml").text.replace('\xa0',' ')
    if 'V O T O' in texto:
        texto = re.split('V O T O', texto)[1]
    # Remover O exmo Juiz Fulano de Tal: no começo do voto
    texto = re.sub('^[OA].{10,140}:', '', texto.strip())
    texto = re.sub('(\n {0, 4})+', '\n', texto.strip())
    texto = remover_citação_de_julgado(texto)
    return texto

def clean_relatorio(texto):
    # Remover cabeçalho e outras coisas antes do voto
    texto = BeautifulSoup(texto, "lxml").text.replace('\xa0',' ')
    if 'R E L A T Ó R I O' in texto: texto = texto.split('R E L A T Ó R I O')[1]
    return texto