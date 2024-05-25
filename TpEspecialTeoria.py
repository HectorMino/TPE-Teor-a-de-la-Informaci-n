import csv
import math
import pandas as pd
import numpy as np
import vectorEstacionario
import huffman

def calcular_media(numeros):
    suma = sum(numeros)
    N = len(numeros)
    return suma / N

def calcular_desvio(numeros, media):
    suma_dif_cuadrada = sum((num - media) ** 2 for num in numeros)
    N = len(numeros)
    return math.sqrt(suma_dif_cuadrada / N)

def leer_csv(filename):
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        datos = [float(row[0]) for row in reader]
    return datos

def categorizar_temperatura(temp):
    if temp < 10:
        return 'B'
    elif 10 <= temp < 20:
        return 'M'
    else:
        return 'A'

def escribir_csv_con_categorias(input_filename, output_filename):
    with open(input_filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        datos = [float(row[0]) for row in reader]
    
    categorias = [categorizar_temperatura(temp) for temp in datos]
    
    with open(output_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for categoria in categorias:
            writer.writerow([categoria])

archivo_csv1 = "S1_buenosAires.csv"
archivo_csv2 = "S2_bogota.csv"  
archivo_csv3 = "S3_vancouver.csv"  

datos1 = leer_csv(archivo_csv1)
datos2 = leer_csv(archivo_csv2)
datos3 = leer_csv(archivo_csv3)

# Calcular media y desvío estándar para cada conjunto de datos
medias = [calcular_media(datos) for datos in [datos1, datos2, datos3]]
desvios = [calcular_desvio(datos, media) for datos, media in zip([datos1, datos2, datos3], medias)]

print("Medias de cada señal:", medias)
print("Desvíos estándar de cada señal:", desvios)

# Función para calcular la correlación cruzada entre dos conjuntos de datos
def calcular_correlacion(senal1, senal2):
    media1 = calcular_media(senal1)
    media2 = calcular_media(senal2)
    
    suma_covarianza = sum((senal1[i] - media1) * (senal2[i] - media2) for i in range(len(senal1)-1))
    suma_varianza1 = sum((senal1[i] - media1) ** 2 for i in range(len(senal1)-1))
    suma_varianza2 = sum((senal2[i] - media2) ** 2 for i in range(len(senal2)-1))
    
    covarianza = suma_covarianza / len(senal1)
    varianza1 = suma_varianza1 / len(senal1)
    varianza2 = suma_varianza2 / len(senal2)
    
    return covarianza / (math.sqrt(varianza1) * math.sqrt(varianza2))

# Calcular la matriz de correlación cruzada
datos = [datos1, datos2, datos3]
num_senales = len(datos)
correlacion = [[0] * num_senales for _ in range(num_senales)]

for i in range(num_senales):
    for j in range(i, num_senales):
        correlacion[i][j] = calcular_correlacion(datos[i], datos[j])
        correlacion[j][i] = correlacion[i][j]

print("correlacion: \n", correlacion)

#EJERCICIO 2:

def getMatrizTransicion(nombre_archivo):
    df = pd.read_csv(nombre_archivo, header=None, names=['estado'])
    estados = ['B', 'M', 'A']
    # Inicializar la matriz de transición con ceros
    transiciones = np.zeros((len(estados), len(estados)))    
    # Crear un diccionario para mapear estados a índices
    estado_a_indice = {estado: i for i, estado in enumerate(estados)}

    for i in range(len(df['estado']) - 1):
        estado_actual = df['estado'][i]
        estado_siguiente = df['estado'][i + 1]
        k = estado_a_indice[estado_actual]
        j = estado_a_indice[estado_siguiente]
        transiciones[k, j] += 1       

    sumaColumn = transiciones.sum(axis=0, keepdims=True)
    # Obtener una lista de índices donde la suma de la columna sea distinta de cero
    indices_no_cero = np.where(sumaColumn != 0)[1]

    # Normalizar solo las columnas que no tengan suma igual a cero
    for i in indices_no_cero:
        transiciones[:, i] /= sumaColumn[0, i]

    # Convertir la matriz a un DataFrame para mejor visualización
    matriz_transicion_visible = pd.DataFrame(transiciones, index=estados, columns=estados)
    return transiciones,matriz_transicion_visible

matrizTransicionBa, matrizTransicionBaVisible = getMatrizTransicion('S1_buenosAires_categorizadas.csv')
matrizTransicionBo, matrizTransicionBoVisible = getMatrizTransicion('S2_bogota_categorizadas.csv')
matrizTransicionVa, matrizTransicionVaVisible = getMatrizTransicion('S3_vancouver_categorizadas.csv')


print("Matriz Transicion Buenos Aires:")
print(matrizTransicionBaVisible)
print("\nMatriz Transicion Bogota:")
print(matrizTransicionBoVisible)
print("\nMatriz Transicion Vancouver:")
print(matrizTransicionVaVisible)
print()



def getMatrizSinMemoria(nombre_archivo):
    matrizSinMemoria = [0, 0, 0]
    df = pd.read_csv(nombre_archivo, header=None, names=['estado'])
    for i in range(len(df['estado'])):
        if df['estado'][i] == 'A':
            matrizSinMemoria[2] += 1
        elif df['estado'][i] == 'M':
            matrizSinMemoria[1] += 1
        else:
            matrizSinMemoria[0] += 1
    for i in range(len(matrizSinMemoria)):
        matrizSinMemoria[i] /= len(df['estado'])  # Normalizar los valores
    return matrizSinMemoria

matrizSinMemoriaBa = getMatrizSinMemoria('S1_buenosAires_categorizadas.csv')
matrizSinMemoriaBo = getMatrizSinMemoria('S2_bogota_categorizadas.csv')
matrizSinMemoriaVa = getMatrizSinMemoria('S3_vancouver_categorizadas.csv')

print("Matriz Sin Memoria Buenos Aires: " + "B: ",matrizSinMemoriaBa[0], " M:",matrizSinMemoriaBa[1], " A:",matrizSinMemoriaBa[2])
print('Matriz Sin Memoria Bogota: ' + "B:",matrizSinMemoriaBo[0] , "M:" , matrizSinMemoriaBo[1] , "A:" , matrizSinMemoriaBo[2])
print('Matriz Sin Memoria Vancouver: ' + "B:" , matrizSinMemoriaVa[0] , "M:" , matrizSinMemoriaVa[1] , "A:" , matrizSinMemoriaVa[2])
print()

def calcularEntropiaH1(matrizSinMemoria):
    entropiaSinMem = 0
    for i in range(len(matrizSinMemoria)):
        if (matrizSinMemoria[i] != 0.0):
            entropiaSinMem += matrizSinMemoria[i] * np.log2(matrizSinMemoria[i])
    return -entropiaSinMem

entropiaSinMem = calcularEntropiaH1(matrizSinMemoriaBa)
print("Entropia sin memoria Buenos Aires: " + str(entropiaSinMem))
entropiaSinMem = calcularEntropiaH1(matrizSinMemoriaBo)
print("\nEntropia sin memoria Bogota: " + str(entropiaSinMem))
entropiaSinMem = calcularEntropiaH1(matrizSinMemoriaVa)
print("\nEntropia sin memoria Vancouver: " + str(entropiaSinMem))

# Definir una función para obtener la matriz acumulada por columnas
def obtener_matriz_acumulada_columnas(matriz_transicion):
    matriz_acumulada = matriz_transicion.cumsum(axis=0)
    return matriz_acumulada

# Obtener la matriz acumulada para cada ciudad
matriz_acumulada_ba = obtener_matriz_acumulada_columnas(matrizTransicionBa)
matriz_acumulada_bo = obtener_matriz_acumulada_columnas(matrizTransicionBo)
matriz_acumulada_va = obtener_matriz_acumulada_columnas(matrizTransicionVa)


print("\nMatriz acumulada para Buenos Aires:")
print(matriz_acumulada_ba)

print("\nMatriz acumulada para Bogotá:")
print(matriz_acumulada_bo)

print("\nMatriz acumulada para Vancouver:")
print(matriz_acumulada_va)


vectorEstacionarioBa = vectorEstacionario.calcular_Vector_Estacionario(matriz_acumulada_ba)
print('Vector Estacionario Buenos Aires: ' + str(vectorEstacionarioBa))
vectorEstacionarioBo = vectorEstacionario.calcular_Vector_Estacionario(matriz_acumulada_bo)
print('Vector Estacionario Bogota: ' + str(vectorEstacionarioBo))
vectorEstacionarioVa = vectorEstacionario.calcular_Vector_Estacionario(matriz_acumulada_va)
print('Vector Estacionario Vancouver: ' + str(vectorEstacionarioVa))

def Hcondicional(matrizTransicion, vectorEst):
    entropiaMem = 0    
    for i in range(len(vectorEst)):
        entropiaMemAux = 0
        for j in range(len(matrizTransicion)):
            if matrizTransicion[j, i] != 0:
                entropiaMemAux += matrizTransicion[j, i] * np.log2(matrizTransicion[j, i])        
        entropiaMem += -entropiaMemAux * vectorEst[i]
    print(entropiaMem)
    return entropiaMem

entropiaConMem = Hcondicional(matrizTransicionBa, vectorEstacionarioBa)
print('Entropia con memoria Ba: ' + str(entropiaConMem))
entropiaConMem = Hcondicional(matrizTransicionBo, vectorEstacionarioBo)
print('Entropia con memoria Bogota: ' + str(entropiaConMem))
entropiaConMem = Hcondicional(matrizTransicionVa, vectorEstacionarioVa) 
print('Entropia con memoria Vancouver: ' + str(entropiaConMem))



def ordenUno(vectorEstacionario):  
    S = [
        {'simbolo': 'B', 'probabilidad': vectorEstacionario[0]},
        {'simbolo': 'M', 'probabilidad': vectorEstacionario[1]},
        {'simbolo': 'A', 'probabilidad': vectorEstacionario[2]}
    ]
    return S




def extensionOrdenDos(vectorEstacionario, matrizTransicion):   
    S = [
        {'simbolo': 'BB', 'probabilidad': vectorEstacionario[0] * matrizTransicion[0,0]},
        {'simbolo': 'BM', 'probabilidad': vectorEstacionario[0] * matrizTransicion[1,0]},
        {'simbolo': 'BA', 'probabilidad': vectorEstacionario[0] * matrizTransicion[2,0]},
        {'simbolo': 'MB', 'probabilidad': vectorEstacionario[1] * matrizTransicion[0,1]},
        {'simbolo': 'MM', 'probabilidad': vectorEstacionario[1] * matrizTransicion[1,1]},
        {'simbolo': 'MA', 'probabilidad': vectorEstacionario[1] * matrizTransicion[2,1]},
        {'simbolo': 'AB', 'probabilidad': vectorEstacionario[2] * matrizTransicion[0,2]},
        {'simbolo': 'AM', 'probabilidad': vectorEstacionario[2] * matrizTransicion[1,2]},
        {'simbolo': 'AA', 'probabilidad': vectorEstacionario[2] * matrizTransicion[2,2]},
    ]    
    return S

def impresionCodigoS1(vectorEstacionario):
    huffman.codigos = {}
    resultado = huffman.huffman(ordenUno(vectorEstacionario))
    return resultado

def impresionCodigoS2(vectorEstacionario, matrizTransicion):
    huffman.codigos = {}
    resultadoDos = huffman.huffman(extensionOrdenDos(vectorEstacionario,matrizTransicion))
    return resultadoDos

def longitudMedia(simbolosProb, codificaciones):
    suma = 0
    long_media = 0
    # Crear un diccionario para acceder rápidamente a las probabilidades por símbolo
    simbolosProbDict = {item['simbolo']: item['probabilidad'] for item in simbolosProb}
    
    for simbolo, codigo in codificaciones.items():   
        suma = len(codigo)    
        # Acceder a la probabilidad usando el diccionario
        long_media = long_media + (suma * simbolosProbDict[simbolo])
    return long_media

def impresionLongitudMedia(vectorEstacionario, matrizTransicion):
    huffman.codigos = {}
    resultado = huffman.huffman(ordenUno(vectorEstacionario))
    huffman.codigos = {}
    resultadoDos = huffman.huffman(extensionOrdenDos(vectorEstacionario,matrizTransicion))
    simbolosOrdenUno = ordenUno(vectorEstacionario)
    simbolosOrdenDos = extensionOrdenDos(vectorEstacionario, matrizTransicion)
    huffman.codigos = {}

    return "Longitud Media: " + str(longitudMedia(simbolosOrdenUno, resultado)) +"\nLongitud Media Orden Dos: "+ str(longitudMedia(simbolosOrdenDos, resultadoDos))


print('\nBuenos Aires: ')
print("Codigos S1")
print(impresionCodigoS1(vectorEstacionarioBa))
print("Codigos S2")
print(impresionCodigoS2(vectorEstacionarioBa, matrizTransicionBa))
print(impresionLongitudMedia(vectorEstacionarioBa, matrizTransicionBa))

print('\nBogota: ')
print("Codigos S1")
print(impresionCodigoS1(vectorEstacionarioBo))
print("Codigos S2")
print(impresionCodigoS2(vectorEstacionarioBo, matrizTransicionBo))
print(impresionLongitudMedia(vectorEstacionarioBo, matrizTransicionBo))

print('\nVancouver: ')
print("Codigos S1")
print(impresionCodigoS1(vectorEstacionarioVa))
print("Codigos S2")
print(impresionCodigoS2(vectorEstacionarioVa, matrizTransicionVa))
print(impresionLongitudMedia(vectorEstacionarioVa, matrizTransicionVa))




def longBitsOrdenUno(nombre_archivo):
    cantB = 0
    cantM = 0
    cantA = 0
    df = pd.read_csv(nombre_archivo, header=None, names=['estado'])   
    for i in range(len(df['estado'])):
        estado_actual = df['estado'][i]
        if (estado_actual == 'B'):
            cantB += 1
        elif(estado_actual == 'M'):
            cantM += 1
        else:
            cantA += 1
    return cantB, cantM, cantA

cantB_BuenosAires,cantM_BuenosAires,cantA_BuenosAires = longBitsOrdenUno('S1_buenosAires_categorizadas.csv')
print('cantB: '+str(cantB_BuenosAires) + ' cantM: '+str(cantM_BuenosAires) + ' cant A: '+str(cantA_BuenosAires))

cantB_Bogota,cantM_Bogota,cantA_Bogota = longBitsOrdenUno('S2_bogota_categorizadas.csv')
print('cantB: '+str(cantB_Bogota) + ' cantM: '+str(cantM_Bogota) + ' cant A: '+str(cantA_Bogota))

cantB_Vancouver,cantM_Vancouver,cantA_Vancouver = longBitsOrdenUno('S3_vancouver_categorizadas.csv')
print('cantB: '+str(cantB_Vancouver) + ' cantM: '+str(cantM_Vancouver) + ' cant A: '+str(cantA_Vancouver))


def longBitsOrdenDos(nombre_archivo):
    cantBB = 0
    cantBM = 0
    cantBA = 0
    cantMB = 0
    cantMM = 0
    cantMA = 0
    cantAB = 0
    cantAM = 0
    cantAA = 0
    
    df = pd.read_csv(nombre_archivo, header=None, names=['estado'])    
    
    for i in range(0, len(df['estado'])-1, 2):
        estado_actual = df['estado'][i]
        estado_siguiente = df['estado'][i+1]   
        
        if estado_actual == 'B':
            if estado_siguiente == 'B':
                cantBB += 1
            elif estado_siguiente == 'M':
                cantBM += 1
            else:
                cantBA += 1
        elif estado_actual == 'M':
            if estado_siguiente == 'B':
                cantMB += 1
            elif estado_siguiente == 'M':
                cantMM += 1
            else:
                cantMA += 1
        else:
            if estado_siguiente == 'B':
                cantAB += 1
            elif estado_siguiente == 'M':
                cantAM += 1
            else:
                cantAA += 1
    
    return cantBB, cantBM, cantBA, cantMB, cantMM, cantMA, cantAB, cantAM, cantAA

cantBB_BuenosAires, cantBM_BuenosAires, cantBA_BuenosAires, cantMB_BuenosAires, cantMM_BuenosAires, cantMA_BuenosAires, cantAB_BuenosAires, cantAM_BuenosAires, cantAA_BuenosAires = longBitsOrdenDos('S1_buenosAires_categorizadas.csv')
print('cantBB:', cantBB_BuenosAires, 'cantBM:', cantBM_BuenosAires, 'cantBA:', cantBA_BuenosAires, 'cantMB:', cantMB_BuenosAires, 'cantMM:', cantMM_BuenosAires, 'cantMA:', cantMA_BuenosAires, 'cantAB:', cantAB_BuenosAires, 'cantAM:', cantAM_BuenosAires, 'cantAA:', cantAA_BuenosAires)

cantBB_Bogota, cantBM_Bogota, cantBA_Bogota, cantMB_Bogota, cantMM_Bogota, cantMA_Bogota, cantAB_Bogota, cantAM_Bogota, cantAA_Bogota = longBitsOrdenDos('S2_bogota_categorizadas.csv')
print('cantBB:', cantBB_Bogota, 'cantBM:', cantBM_Bogota, 'cantBA:', cantBA_Bogota, 'cantMB:', cantMB_Bogota, 'cantMM:', cantMM_Bogota, 'cantMA:', cantMA_Bogota, 'cantAB:', cantAB_Bogota, 'cantAM:', cantAM_Bogota, 'cantAA:', cantAA_Bogota)

cantBB_Vancouver, cantBM_Vancouver, cantBA_Vancouver, cantMB_Vancouver, cantMM_Vancouver, cantMA_Vancouver, cantAB_Vancouver, cantAM_Vancouver, cantAA_Vancouver = longBitsOrdenDos('S3_vancouver_categorizadas.csv')
print('cantBB:', cantBB_Vancouver, 'cantBM:', cantBM_Vancouver, 'cantBA:', cantBA_Vancouver, 'cantMB:', cantMB_Vancouver, 'cantMM:', cantMM_Vancouver, 'cantMA:', cantMA_Vancouver, 'cantAB:', cantAB_Vancouver, 'cantAM:', cantAM_Vancouver, 'cantAA:', cantAA_Vancouver)

def teoremaShannonSinMemoria(logMedia,cantMuestra,h1):
    print(h1, ' <= ', logMedia/cantMuestra, ' < ', h1 + 1/cantMuestra)
    if(h1<logMedia/cantMuestra<h1+1/cantMuestra):
        return True
    return False

def teoremaShannonMemoria(hCond,logMedia,cantMuestra,h1):
    desigualdadIzq= h1/cantMuestra+(1-1/cantMuestra)*hCond
    desigualdadDer=h1/cantMuestra+(1-1/cantMuestra)*hCond+1/cantMuestra
    print(desigualdadIzq, ' <= ', logMedia/cantMuestra, ' < ', desigualdadDer)
    if(desigualdadIzq<logMedia/cantMuestra<desigualdadDer):
        return True
    return False
        
#---------------------------CORROBORANDO SHANNON SIN MEMORIA------------------------#
print("Shannon sin memoria")
print("Shannon Buenos aires")
print(teoremaShannonSinMemoria(longitudMedia(ordenUno(vectorEstacionarioBa), huffman.huffman(ordenUno(vectorEstacionarioBa))) ,1,calcularEntropiaH1(matrizSinMemoriaBa)))      

print("\nShannon Bogota")
huffman.codigos = {}
print(teoremaShannonSinMemoria(longitudMedia(ordenUno(vectorEstacionarioBo), huffman.huffman(ordenUno(vectorEstacionarioBo))) ,1,calcularEntropiaH1(matrizSinMemoriaBo))) 

print("\nShannon Vancouver")
huffman.codigos = {}

print(teoremaShannonSinMemoria(longitudMedia(ordenUno(vectorEstacionarioVa), huffman.huffman(ordenUno(vectorEstacionarioVa))) ,1,calcularEntropiaH1(matrizSinMemoriaVa)))
print('\n')
#--------------------------------CORROBORANDO SHANNON CON MEMORIA (y orden 2)-----------------------
print("Shannon con memoria")
print("Shannon Buenos aires")
huffman.codigos = {}
print(teoremaShannonMemoria(Hcondicional(matrizTransicionBa, vectorEstacionarioBa),longitudMedia(extensionOrdenDos(vectorEstacionarioBa,matrizTransicionBa), huffman.huffman(extensionOrdenDos(vectorEstacionarioBa,matrizTransicionBa)))  ,2,calcularEntropiaH1(matrizSinMemoriaBa)))      

print("\nShannon Bogota")
huffman.codigos = {}
print(teoremaShannonMemoria(Hcondicional(matrizTransicionBo, vectorEstacionarioBo),longitudMedia(extensionOrdenDos(vectorEstacionarioBo,matrizTransicionBo), huffman.huffman(extensionOrdenDos(vectorEstacionarioBo,matrizTransicionBo))),2,calcularEntropiaH1(matrizSinMemoriaBo))) 

print("\nShannon Vancouver")
huffman.codigos = {}
print(teoremaShannonMemoria(Hcondicional(matrizTransicionVa, vectorEstacionarioVa),longitudMedia(extensionOrdenDos(vectorEstacionarioVa,matrizTransicionVa), huffman.huffman(extensionOrdenDos(vectorEstacionarioVa,matrizTransicionVa))) ,2,calcularEntropiaH1(matrizSinMemoriaVa)))
print('\n')