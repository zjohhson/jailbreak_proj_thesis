from language_models import GPT
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_embedding(text):
  embedding_model = GPT("text-embedding-ada-002") # needs to be updated if we want to use other embedding models
  try:
   text = text.replace("\n", " ")
   return embedding_model.client.embeddings.create(input = [text], model="text-embedding-ada-002").data[0].embedding
  except Exception as error:
    print(text, error)

def get_vecs(data_path):
    # Import dataset of labeled responses to harmful requests (1 = compliant, 0 = avoidant/evasive)
    eval_tuning_df = pd.read_csv(data_path)
    eval_tuning_df['ada_embedding'] = eval_tuning_df['response'].apply(lambda x: get_embedding(x))

    ## Uncomment if we want to save embedding vectors ##
    # eval_tuning_df.to_csv('embeddings.csv', index=False) 

    # Convert embedding vectors to numpy float32 arrays for PCA and SVM
    eval_tuning_df['ada_embedding'] = eval_tuning_df['ada_embedding'].apply(lambda x: np.array(x, dtype=np.float32))
    vecs = []
    for vec in eval_tuning_df['ada_embedding']:
        vecs.append(vec)
    
    return vecs

def get_labels(data_path):
    eval_tuning_df = pd.read_csv(data_path)
    return list(eval_tuning_df['label'])

# # Creates PCA model that is fitted to vecs
# def create_pca(vecs):
#     x_PCA = StandardScaler().fit_transform(np.array(vecs))
#     PCA_ = PCA(n_components=2)
#     principalComponents = PCA_.fit_transform(x_PCA)
#     principalDf = pd.DataFrame(data = principalComponents
#              , columns = ['principal component 1', 'principal component 2'])
#     xPCA = list(principalDf['principal component 1'])
#     yPCA = list(principalDf['principal component 2'])

#     return (xPCA, yPCA, PCA_)

# # Applies already-fitted PCA model to new data
# def apply_pca(pca, embedding):
#     embedding_arr = np.array([embedding])
#     coords = pca.transform([embedding])[0]
#     x, y = coords[0], coords[1]

#     return x, y

def do_pca(vecs):
    scaler = StandardScaler()
    scaler.fit(np.array(vecs))
    scaled = scaler.transform(np.array(vecs))
    pca = PCA(n_components = 2)
    pca.fit(scaled)
    return pca, scaler

def get_svm_classifier(xPCA, yPCA, labels):
    X = []
    Y = []
    for i in range(len(xPCA)):
        X.append([xPCA[i], yPCA[i]])
        Y.append(labels[i])

    model = SVC(kernel='linear', C=1E10)
    model.fit(X,Y)
    
    return (model, X, Y)

def plot_svc_decision_function(model, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])

    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

# Scatterplot of PCA embeddings of responses with SVM linear boundary
def plot_svm_boundary(xPCA, yPCA, labels):
    # Colors for scatterplot, can be changed
    colormap = {
       0: 'blue',
       1: 'red'
    }

    (svm, X, Y) = get_svm_classifier(xPCA, yPCA, labels)

    # coordinates for non-jailbroken responses (label 0)
    x_plot_0 = []
    y_plot_0 = []
    c_plot_0 = []

    x_plot_1 = []
    y_plot_1 = []
    c_plot_1 = []

    for i in range(len(X)):
        if labels[i]:
            x_plot_1.append(X[i][0])
            y_plot_1.append(X[i][1])
            c_plot_1.append(colormap[labels[i]])
        else:
            x_plot_0.append(X[i][0])
            y_plot_0.append(X[i][1])
            c_plot_0.append(colormap[labels[i]])

    plt.scatter(x_plot_0, y_plot_0, c=c_plot_0, s=50, cmap='autumn', label = 'Non-jailbroken responses')
    plt.scatter(x_plot_1, y_plot_1, c=c_plot_1, s=50, cmap='autumn', label = 'Jailbroken responses')

    plot_svc_decision_function(svm)

    plt.title('PCA on embeddings of jailbroken & non-jailbroken ChatGPT responses')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()

    plt.savefig('../data/pca_classification.png')