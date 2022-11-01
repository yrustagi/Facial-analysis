from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

def load_and_center_dataset(filename):
    x = np.load(filename)
    centred = x - np.mean(x,axis=0)
    return centred

def get_covariance(dataset):
    x = np.array(dataset)
    transpose = np.transpose(dataset)
    dot2 = np.dot(transpose, x)
    return dot2/(len(dataset)-1)

def get_eig(S, m):
    x = len(S) - m
    y = len(S) - 1
    Lambda, U = eigh(S, eigvals = (x, y))
    eig_val_matrix = np.diag(Lambda[0:][::-1])
    vec = U[0:,(sorted(range(len(Lambda)))[::-1])]
    return eig_val_matrix,vec

def get_eig_prop(S, prop):
    Lambda, U = eigh(S)
    e_val = Lambda[np.flip((sorted(range(len(Lambda)))))]
    e_vec = U[:, np.flip((sorted(range(len(Lambda)))))]
    for i in e_val:
        j = np.flatnonzero(e_val==i)
        if i/(np.sum(e_val)) <= prop:
            e_val = e_val[e_val != i]
            e_vec = np.delete(e_vec, j, axis = 1)
    return np.diag(e_val), e_vec

def project_image(image, U):
    x = np.dot(np.transpose(U), image)
    output = np.dot(U, x)
    return output

def display_image(orig, proj):
    original = orig.reshape((32, 32), order = "F")
    projection = proj.reshape((32, 32), order = "F")

    figure, (x, y) = plt.subplots(nrows = 1, ncols = 2)

    x.set_title('Orignal')
    y.set_title('Projection')

    old = x.imshow(original, aspect='equal')
    new = y.imshow(projection, aspect='equal')

    figure.colorbar(old, ax=x)
    figure.colorbar(new, ax=y)
    
    plt.show()
