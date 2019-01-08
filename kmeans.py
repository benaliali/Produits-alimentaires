import pandas as pd
import numpy as np
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler 
from matplotlib.collections import LineCollection
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


def display_circles(pcs, n_comp, pca, axis_ranks, labels=None, label_rotation=0, lims=None):
    for d1, d2 in axis_ranks: # On affiche les 3 premiers plans factoriels, donc les 6 premières composantes
        if d2 < n_comp:

            # initialisation de la figure
            fig, ax = plt.subplots(figsize=(7,6))

            # détermination des limites du graphique
            if lims is not None :
                xmin, xmax, ymin, ymax = lims
            elif pcs.shape[1] < 30 :
                xmin, xmax, ymin, ymax = -1, 1, -1, 1
            else :
                xmin, xmax, ymin, ymax = min(pcs[d1,:]), max(pcs[d1,:]), min(pcs[d2,:]), max(pcs[d2,:])

            # affichage des flèches
            # s'il y a plus de 30 flèches, on n'affiche pas le triangle à leur extrémité
            if pcs.shape[1] < 30 :
                plt.quiver(np.zeros(pcs.shape[1]), np.zeros(pcs.shape[1]),
                   pcs[d1,:], pcs[d2,:], 
                   angles='xy', scale_units='xy', scale=1, color="grey")
                # (voir la doc : https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html)
            else:
                lines = [[[0,0],[x,y]] for x,y in pcs[[d1,d2]].T]
                ax.add_collection(LineCollection(lines, axes=ax, alpha=.1, color='black'))
            
            # affichage des noms des variables  
            if labels is not None:  
                for i,(x, y) in enumerate(pcs[[d1,d2]].T):
                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax :
                        plt.text(x, y, labels[i], fontsize='14', ha='center', va='center', rotation=label_rotation, color="blue", alpha=0.5)
            
            # affichage du cercle
            circle = plt.Circle((0,0), 1, facecolor='none', edgecolor='b')
            plt.gca().add_artist(circle)

            # définition des limites du graphique
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
        
            # affichage des lignes horizontales et verticales
            plt.plot([-1, 1], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-1, 1], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Cercle des corrélations (F{} et F{})".format(d1+1, d2+1))
            plt.show(block=False)
        
def display_factorial_planes(X_projected, n_comp, pca, axis_ranks, labels=None, alpha=1, illustrative_var=None):
    for d1,d2 in axis_ranks:
        if d2 < n_comp:
 
        
            # affichage des points
            if illustrative_var is None:
                plt.scatter(X_projected[:, d1], X_projected[:, d2], alpha=alpha)
            else:
                illustrative_var = np.array(illustrative_var)
                for value in np.unique(illustrative_var):
                    selected = np.where(illustrative_var == value)
                    plt.scatter(X_projected[selected, d1], X_projected[selected, d2], alpha=alpha, label=value)
                plt.legend()

            # affichage des labels des points
            if labels is not None:
                for i,(x,y) in enumerate(X_projected[:,[d1,d2]]):
                    plt.text(x, y, labels[i],
                              fontsize='14', ha='center',va='center') 
                
            # détermination des limites du graphique
            boundary = np.max(np.abs(X_projected[:, [d1,d2]])) * 1.1
            plt.xlim([-boundary,boundary])
            plt.ylim([-boundary,boundary])
        
            # affichage des lignes horizontales et verticales
            plt.plot([-100, 100], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-100, 100], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Projection des individus (sur F{} et F{})".format(d1+1, d2+1))
            plt.show(block=False)

def display_scree_plot(pca):
    scree = pca.explained_variance_ratio_*100
    plt.bar(np.arange(len(scree))+1, scree)
    plt.plot(np.arange(len(scree))+1, scree.cumsum(),c="red",marker='o')
    plt.xlabel("rang de l'axe d'inertie")
    plt.ylabel("pourcentage d'inertie")
    plt.title("Eboulis des valeurs propres")
    plt.show(block=False)



# préparation des données  
# vérification de dataset si contient des valeurs manquantes
dataFrame = pd.read_excel("data.xlsx",index_col=0)

sugars=dataFrame.loc[~dataFrame["Sugars"].isnull(),["Sugars","Dietary Fiber" ]]  # ~ = not
#dimensions des  données
dataFrame.shape#61 portions alimentaire et 16 variables

#statistiques descriptives 
dataFrame.describe()

# booleen qui indique si il y'a des valeurs manquantes 
nan_verifiation = dataFrame.isnull().values.any()

# toutes les colomuns qui contiennent des valeurs manquantes 
col_val_manq = dataFrame.columns[dataFrame.isna().any()].tolist()

# Suppression des colomns qui contiennent des valeurs manquentes
dataFrame = dataFrame.drop(columns=col_val_manq) 

#toutes les columns du data frame 
col = dataFrame.columns

#toutes les index 
index_ = dataFrame.index

# Changer l'echelle des varaibles (pour equilibrer les données)
sc = StandardScaler() 
dataFrame = pd.DataFrame(sc.fit_transform(dataFrame))



#########################
sugars=pd.DataFrame(sugars)
col_sug=sugars.columns
inex_sug = sugars.index
sugars = pd.DataFrame(sc.fit_transform(sugars))
sugars.columns = col_sug
sugars.index = inex_sug

##########################
# remetre les les noms des index et noms des columns a leur place 
dataFrame.columns = col
dataFrame.index = index_

# visualisation des correlations 
scatter_matrix(dataFrame, figsize=(9, 9))
plt.show()
# Total Carbo-Hydrate ,Calories semble corréle pour s'assurer en utilise la fonctions corr()


# tableau de corrélation en generale deux variabeles correlées si leux taux de corrélation depasse 0.75
tab_corr = dataFrame.corr() # Total Carbo-Hydrate ,Calories


# ACP (Analyse en composentes prinicpales) 
pca = PCA(n_components=2).fit(dataFrame)
x_reduit = pca.transform(dataFrame)


# Analyser les performances de la acp 
# valeurs propres via la fonction display_scree_plot,
display_scree_plot(pca)#selon le graph avec seulement 2 composentes on a plus de 90% de l'informations 


# Cercle des corrélations
pcs = pca.components_
display_circles(pcs, 2, pca, [(0,1)], labels = dataFrame.columns)


# Projection des individus selon les deux composantes principal obtenus 
display_factorial_planes(x_reduit, 2, pca, [(0,1)], labels = dataFrame.index)


#pour determiner le nombre de cluster optimal deux methodes sont utiliser Elbow et silhouette
# Elbow 
wcss = []
for i in range(2,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)
    kmeans.fit(x_reduit)
    wcss.append(kmeans.inertia_)

plt.plot(range(2, 11), wcss)
plt.title('Elbow')
plt.xlabel('Num de cluster')
plt.ylabel('WCSS')
plt.show()  


# silhouette_score 
#utilisation de la métric silhoute pour determiner le nombre optimale de cluster 
res=[]
for i in range(2,11):
    
    k_means=KMeans(n_clusters=i,random_state=1)#le random state a été fixé pour avoir le meme resultats achaque execution
    k_means.fit(x_reduit)
    res.append(silhouette_score(x_reduit,k_means.labels_))
    
#graphique
plt.title('silhouette')
plt.xlabel('num de cluster')
plt.plot(np.arange(2,11),res)
plt.show()


#selon les deux methodes nombre de cluster optimale est k = 3

# Kmeans
kmeans=KMeans(n_clusters=3,random_state=1)
kmeans.fit(dataFrame)
dataFrame_new=dataFrame.copy()
# je crée la colonne kmeans dans le dataframe créé au dessus
dataFrame_new.loc[:,'kmeans'] = kmeans.labels_

# avec les classes des kmeans
display_factorial_planes(x_reduit, 2, pca, [(0,1)], illustrative_var = kmeans.labels_)

# classe 0 : fruits de mer
# classe 1 : fruits et legumes
# classe 2 : poissons 

