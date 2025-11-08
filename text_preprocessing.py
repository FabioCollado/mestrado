from tensorflow.keras.preprocessing.text import text_to_word_sequence
import re
import itertools

def remove_header(text):
    # A estratégia é remover parágrafos repetidos
    # Dificilmente haverá parágrafos repetidos que não sejam cabeçalhos
    # Mais difícil ainda é que um parágrafo repetido que não seja cabeçalho traga alguma informação útil
    
    splitted_new_text = []
    for paragraph in text.split('\n\n'):
        if paragraph not in splitted_new_text: splitted_new_text.append(paragraph)
    return '\n\n'.join(splitted_new_text)

# Preprocessa um texto com o objetivo específico de separar em parágrafos utilizando o '.' como delimitador.
def preprocessaeFiltra(texto):
    # itens que começam linhas são bons separadores
    texto = remove_header(texto)
    texto = re.sub("""\\n\\n(?= ?[“"']?[a-z] ?[\)\-])""", ".", texto, flags=re.IGNORECASE)

    #\n\nRua João Ebóli, n° 100, Apto 183, Torre 2, Vila Planalto, São Bernardo Do Campo, São Paulo, CEP: 09895-550, Tel (11) 98222-8851 \n\n(Email) elismarcia.com@hotmail.com  \n\n ADVOCACIA  ELISANGELA MARCIA DOS SANTOS   \n\n
    #\n\nAvenida Dom Pedro II, nº 301 – centro – Sala 52 – 5 º Andar  \n\n  7 \n\nROSANA ANANIAS LINO \n\nADVOGADA – OAB/SP 265.496 \n\n
    
    texto = texto.replace("\t"," ")
    texto = texto.replace("\xa0"," ")
    texto = texto.replace(".º", "º")
    texto = texto.replace(".ª", "ª")
    texto = texto.replace('\n', ' ')
    
    # ==================================Daqui em diante, \n foi removido==============================
    #Como o tokenizador separa por ' ', preciso garantir que ',', ')' e '(' sejam separados corretamente.
    texto = texto.replace(',', ' , ').replace(')', ' ) ').replace('(', ' ( ').replace(';', ' ; ').replace('-', ' - ')
    
    # Mais do que três . normalmente é marcador de linha
    # Vou retirar ... também porque frequentemente é utilizado sem interromper uma ideia
    texto = re.sub("\.\.\.+", " ", texto, flags=re.IGNORECASE)
    texto = re.sub("\([\. ]*…?\)", " ", texto, flags=re.IGNORECASE)
    
    
    # os autores(as)
    texto = re.sub("\(a?s?\)", " ", texto, flags=re.IGNORECASE)
    # Remove 'http://www.'
    texto = re.sub("(https?:\/\/)?www\.", "", texto, flags=re.IGNORECASE)
    
    # Remove '.com.br','.com'
    texto = re.sub("\.com(\.br)?", "", texto, flags=re.IGNORECASE)

    # Remove '.com.br','.com'
    texto = re.sub("\.pdf", "pdf", texto, flags=re.IGNORECASE)
    
    # Remove '.jus.br'
    texto = re.sub("\.jus(\.br)?", "", texto, flags=re.IGNORECASE)

    # Remove '.jus.br'
    texto = re.sub("\.org(\.br)?", "", texto, flags=re.IGNORECASE)
    
    # Remove '.net'
    texto = re.sub("\.net", "", texto, flags=re.IGNORECASE)
    
    # Remove 'Artigo 44 e ss. do CPC'
    texto = re.sub("s\.?s\.", " seguintes ", texto, flags=re.IGNORECASE)
    
    # Remove página 33 do meio do texto
    texto = re.sub("p[aá]g(ina)?\.? ?[0-9]+ ?(de [0-9]*)?", "", texto, flags=re.IGNORECASE)
    
    
    
    #'r. Decisum', 'p. r. i.', 'c. STJ', 'v. acórdão', '6º T.', 'd. Joaquim da Silva', 'j. 23/11/2020', 'esse M. Juízo'
    texto = re.sub("(?<=[ \.])[privctdjm]\.", "", texto, flags=re.IGNORECASE)
    
    #Assinado eletronicamente por RONILCE MARTINS MACIEL DE OLIVEIRA 09 03 2016 00 06 21 num 50241 https pje1g. trf3 443 pje Processo ConsultaDocumento listView. seam x 16030900062187000000000049442 Número do documento 16030900062187000000000049442 
    #Assinado eletronicamente por assinatura modo teste 05 04 2016 16 55 08 num 87633 TR http pjetg. trf3 jus. bripje Processo ConsultaDocumentolistView. seam x 16040516550800000000000086421 Fica Número do documento 16040516550800000000000086421
    texto = re.sub("Assinado eletronicamente por.{0,1000}listView.{0,1000}Número do documento: [0-9]*", "", texto, flags=re.IGNORECASE)
    
    remove_dot_list = ['pág', ' pag', ' des', ' min', ' e', ' c', 'exa', ' exmo', ' exma', ' del', ' rel', 'res', 'ltda', 'número', ' núm', ' num', ' n', 'art', 'arts', 'artigo', 'artigos', 'inc', ' resp', 'rext', ' cep', 'proc', 'º', 'ª', ' ap', ' sr', ' dr', ' fl', ' fls', ' $', ' apt', ' cj', ' tel', ' av', ' ed', 'julg', ' adv', ' pq', ' conj', ' doc', ' docs', ' cf', ' eg', ' prof', ' dec', ' ec', ' del', ' dl', ' in', ' ac', ' Fax', ' ns']
    for word in remove_dot_list:
        texto = re.sub('(?<=' + word + ')\.', ' ', texto, flags=re.IGNORECASE)
    
    #Remove <...........>
    texto = re.sub("<.{0,60}?>", " ", texto, count=0, flags=0)
    
    texto = re.sub("s\.a\.", "SA", texto, flags=re.IGNORECASE)
    
    #Remove '.' de 'Lei n 11.941', 'Processo n 1223222-20.8.26.0000'
    #Remove '-' de 'Processo n 1223222-20.8.26.0000', 'CPF/MF n.º 284.996.033-00'
    texto = re.sub("(?<=[0-9])\.(?=[0-9])", "", texto, count=0, flags=0)
    texto = re.sub("(?<=[0-9])\-(?=[0-9])", "", texto, count=0, flags=0)
    #texto = re.sub("\. *(?=[A-ZÁÀÂÃÉÈÊÍÏÓÔÕÖÚÇÑ])", " ", texto, count=0, flags=0)

    #texto = filtra(texto)
    #condeno o INSS ao pagamento de honorários advocatícios, que os fixo em R$ 1.000,00
    
    #Remove múltiplos espaços
    texto = re.sub("  +", " ", texto)
    
    return texto.strip()


#===========================================================================================
#===============================Split Paragraphs============================================
#===========================================================================================


def split_list(lista:list, delimiter:str):
    return [list(y) for x, y in itertools.groupby(lista, lambda q: q == delimiter) if not x]

def split_paragraph_again(splitted_paragraphs, pattern, max_size):
    new_splitted_paragraphs = []
    for splitted_paragraph in splitted_paragraphs:
        if len(splitted_paragraph) > max_size:
            parts_of_paragraph = split_list(splitted_paragraph, pattern)
            for part in parts_of_paragraph[:-1]:
                new_splitted_paragraphs.append(part + [';'])
            new_splitted_paragraphs.append(parts_of_paragraph[-1])
        else:
            new_splitted_paragraphs.append(splitted_paragraph)
    return new_splitted_paragraphs


def split_paragraphs(text, max_size = 200, debug = False, lower=True):
    # try to break paragraphs at the correct point:
    preprocessed_text = preprocessaeFiltra(text)
    paragraphs = [p + '.' for p in preprocessed_text.split('.') if p]
    
    splitted_paragraphs = [text_to_word_sequence(p, filters='!"#$%&*+/:=?@[\\]^_`{|}~\t\n', lower=lower) for p in paragraphs]
    
    # Parágrafos que possuem mais que max_size precisam ser quebrados por ;
    splitted_paragraphs = split_paragraph_again(splitted_paragraphs, ';', max_size)
    
    # break paragraphs in small_paragraphs
    # small_paragraphs are paragraphs that respect max_size
    small_paragraphs = []
    if debug: count_break = 0
    for splitted_paragraph in splitted_paragraphs:
        if debug: count_break -= 1
        for i in range(0, len(splitted_paragraph), max_size):
            small_paragraphs.append(splitted_paragraph[i:i+max_size])
            if debug: count_break += 1
    if debug: print('small_paragraphs:', len(small_paragraphs))

    # merge very small paragraphs into bigger ones, if still respect max_size:
    agglutinated_paragraph = []
    agglutinated_output = []
    if debug: 
        count_empty_paragraphs = 0
        count_agglutinated = -1
    for splitted_paragraph in small_paragraphs:
        if splitted_paragraph_is_valid(splitted_paragraph):
            if debug: assert len(splitted_paragraph) <= max_size
            if len(agglutinated_paragraph) + len(splitted_paragraph) <= max_size:
                agglutinated_paragraph = agglutinated_paragraph + splitted_paragraph
                if debug: count_agglutinated += 1
            else:
                agglutinated_output.append(' '.join(agglutinated_paragraph))
                agglutinated_paragraph = splitted_paragraph
        else:
            if debug: count_empty_paragraphs += 1
    agglutinated_output.append(' '.join(agglutinated_paragraph))
    if debug:
        print('Number of paragraphs:', len(agglutinated_output))
        print('count_break:', count_break) 
        print('count_agglutinated:', count_agglutinated)
        print('count_empty_paragraphs:', count_empty_paragraphs)
        assert count_agglutinated + len(agglutinated_output) + count_empty_paragraphs == len(small_paragraphs)
    
    return agglutinated_output

def splitted_paragraph_is_valid(splitted_paragraph):
    for w in splitted_paragraph:
        if re.findall('[a-z]', w, flags=re.IGNORECASE) != []:
            return True
    return False
