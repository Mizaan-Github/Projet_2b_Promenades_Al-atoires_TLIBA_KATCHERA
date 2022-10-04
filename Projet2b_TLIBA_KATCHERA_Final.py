# Bibliothèques utilisées

import decimal #Bibliothèque decimal
import numpy as np #Bibliothèque numpy
import matplotlib.pyplot as plt #Bibliothèque matplotlib (pour le graphique)

# Vecteur largeur entre les portes

L12 = 125 #Largeur pièce 1-->2
L23 = 100 #Largeur pièce 2 -->3
L34 = 75 #Largeur pièce 3 -->4
L45 = 75 #Largeur pièce 4 --> 5
L51 = 75 #Largeur pièce 5-->1

# Nombre de pieces/salles

n = 5

#Nombre de mouches par salle (conditions initiales) (sans unité)

s1 = 50 #salle1
s2 = 0 #salle2
s3 = 0 #salle3
s4 = 0 #salle4
s5 = 0 #salle5

# Vecteur repartition initiale des mouches

M = np.array([s1,s2,s3,s4,s5])

# Matrice de transition du processus

T = np.array([[0,(L12)/(L12 + L23),0,0,(L51)/(L51+L45)],[(L12)/(L12+L51),0,(L23)/(L23+L34),0,0],[0,(L23)/(L12+L23),0,(L34)/(L45+L34),0],[0,0,(L34)/(L34+L23),0,(L45)/(L45+L51)],[(L51)/(L12+L51),0,0,(L45)/(L34+L45),0]])
A=T

# Valeur initiale de la valeur propre

z = 0.78

#Algorithme de la puissance appliquée à l'inverse

def puissanceInverse(A, z):
    precision = 1e-6
    r = abs(decimal.Decimal(str(precision)).as_tuple().exponent) 
    I = np.identity(n)
    try:
        R = np.linalg.inv(A - z * I)
        q = z
        x = M
        conv = False
        variation = dict()
        variation[q] = x / np.linalg.norm(x)
        while not conv: #Plusieurs itérations de la seconde à la première valeur propre pour les valeurs en fonction du temps
            y = np.dot(R, x)
            x = y / np.linalg.norm(y)
            prec_q = q
            q = np.dot(x, np.dot(A, x))
            variation[q] = np.around(x, r)
            cauchy = np.linalg.norm(q - prec_q)
            conv = cauchy <= precision
        return q, x, variation
    except np.linalg.LinAlgError: #Si z fixé valeur propre retourne erreur
        print(z, 'est valeur propre')
        raise
 
#Probabilité de présence traduit numériquement, fonction numerise.
 
def numerise(x):
    return np.sum(M) / np.sum(x) * x

#Construction graphe variation du nombre de mouches dans le temps 

def construitVariation(variation):
    plt.figure()
    legende = []
    for i in range(n):
        variation_salle_i = []
        for q in variation:
            variation_salle_i.append(numerise(variation[q])[i])
        plt.plot(variation_salle_i)
        legende.append('Piece ' + str(i + 1))
    plt.legend(legende)
    plt.title('Variation du nombre de mouche par piece\n' +
              'init = ' + str(M) + ' n = ' + str(np.sum(M)) + ' z = ' + str(z))
    plt.xlabel('Itération')
    plt.ylabel('Nombre de mouches')

#Affichage des résultats et du graphe dans le temps.

def AffichageResultats(q,variation,x):
    print('\nRegime stationaire :\n', 'Vecteur propre correspondant à la valeur propre la plus grande', q, 'est :\n', x,)
    print('Taux de répartition des mouches en régime stationnaire :',numerise(x))
    print('\nVariation au cours du temps :')
    print ('Le taux de répartition des mouches de départ correpondant à la valeur propre initial q')
    print("x est la taux de répartition des mouches correspondant à la valeur propre q l'appartenant")
    for q in variation:
        x = variation[q]
        print('q =', q, '=> x =',numerise(x))
    construitVariation(variation)
    plt.show()

print('Matrice de transition du processus :\n', A)

#Si valeur propre = z fixé --> retourne erreur

try:
    q, variation, x = puissanceInverse(A, z)
    AffichageResultats(q, x, variation)
except np.linalg.LinAlgError:
    print('le z fixé est déjà une valeur propre') 
