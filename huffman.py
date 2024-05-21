# Clase Nodo para crear los nodos del árbol de Huffman
class Nodo:
    def __init__(self, probabilidad, simbolo, izquierda=None, derecha=None):
        self.probabilidad = probabilidad
        self.simbolo = simbolo
        self.izquierda = izquierda
        self.derecha = derecha
        self.code = ''

# Función para calcular los códigos de Huffman
def calculateCodes(nodo, val=''):
    value = val + nodo.code
    if nodo.izquierda:
        calculateCodes(nodo.izquierda, value)
    if nodo.derecha:
        calculateCodes(nodo.derecha, value)
    if not nodo.izquierda and not nodo.derecha:
        codigos[nodo.simbolo] = value
    return codigos

# Función principal del algoritmo de Huffman
def huffman(S):
    nodos = []
    
    # Creación de nodos e insertado
    for simbolo in S:
        if (simbolo['probabilidad'] != 0):     #If para descartar nodos cuya probabilidad es cero (Caso bogota)
            nodo = Nodo(simbolo['probabilidad'], simbolo['simbolo'])
            nodos.append(nodo)
    
    # Mientras haya más de un nodo en la lista de nodos
    while len(nodos) > 1:
        # Ordenar todos los nodos en orden ascendente según su probabilidad
        nodos = sorted(nodos, key=lambda x: x.probabilidad)
        
        # Tomar los dos nodos con menor probabilidad
        izquierda = nodos[0]
        derecha = nodos[1]
        
        # Asignar código 0 al nodo izquierdo y código 1 al nodo derecho
        izquierda.code = '0'
        derecha.code = '1'
        
        # Combinar los dos nodos más pequeños para crear un nuevo nodo
        nuevo_nodo = Nodo(izquierda.probabilidad + derecha.probabilidad, izquierda.simbolo + derecha.simbolo, izquierda, derecha)
        
        # Eliminar los nodos combinados de la lista de nodos y agregar el nuevo nodo
        nodos = nodos[2:]
        nodos.append(nuevo_nodo)
    
    # El algoritmo finaliza cuando se llega al nodo raíz con probabilidad 1
    return calculateCodes(nodos[0])

# Arreglo que almacena el árbol de Huffman resultante
codigos = {}

