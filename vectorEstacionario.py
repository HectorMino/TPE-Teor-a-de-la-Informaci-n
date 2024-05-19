import random
epsilon = 0.00005
N_Pruebas_Min = 100000


def primer_Simbolo ():
    r=random.random()
    i = 0
    v0Acum = [1/2,3/4,1]
    for i in range (3):
        if (r < v0Acum [i]):
            return i
        
def sig_dado_Ant (s_ant, mAcum):
    r=random.random()
    i = 0
    for i in range (3):
        
        if ( r < mAcum[i][s_ant] ):
            return i 
    return 1

def converge (A, B) -> bool:
    i = 0
    for i in range (3):
        if (abs(A[i] - B[i]) > epsilon ):
           return False 
    return True

def calcular_Vector_Estacionario (mAcum):
   
    emisiones = [0,0, 0] #cantidad de emisiones de cada si 
    vEstacionario = [0,0, 0]   # Vector estacionario actual
    vEstacionarioAnt = [-1,-1, -1] # Vector estacionario anterior
    cant_simb_generados = 0  #cantidad de sÃmbolos generados
    s=primer_Simbolo()
    while (cant_simb_generados < N_Pruebas_Min) or not converge (vEstacionario, vEstacionarioAnt): 
        s = sig_dado_Ant(s,mAcum)
        emisiones[s]=emisiones[s]+1
        cant_simb_generados+=1
        i = 0
        for i in range (3):
            vEstacionarioAnt[i] = vEstacionario[i]
            vEstacionario[i] = emisiones[i]/cant_simb_generados
           
    return vEstacionario 
