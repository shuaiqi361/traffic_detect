import numpy as np
import time
import sys
import itertools
from math import sqrt, ceil
from scipy import linalg
from sklearn.utils.extmath import randomized_svd, row_norms
from sklearn.utils import check_array, check_random_state, gen_even_slices, gen_batches, shuffle


def soft_thresholding(x, lm):
    return np.sign(x) * np.maximum(np.abs(x) - lm, 0.)


def fast_ista(b, A, lmbda, max_iter):
    """
    objective function:
    min: {L2_norm(Ax - b) + L1_norm(x)}
    :param A: Dictionary, with shape: [n_coeffs, n_features]
    :param b: input data with shape: [n_samples, n_features]
    :param lmbda: panelty term for sparsity
    :param max_iter:
    :return: sparse codes with shape: [n_samples, n_coeffs]
    """
    n_coeffs, n_feats = A.shape
    n_samples = b.shape[0]
    x = np.zeros(shape=(n_samples, n_coeffs))
    losses = []
    t = 1.
    z = x.copy()  # n_samples, n_coeffs
    L = linalg.norm(A, ord=2) ** 2  # Lipschitz constant

    for k in range(max_iter):
        xold = x.copy()
        z = z + np.dot(b - z.dot(A), A.T) / L
        x = soft_thresholding(z, lmbda / L)
        t0 = t
        t = (1. + sqrt(1. + 4. * t ** 2)) / 2.
        z = x + ((t0 - 1.) / t) * (x - xold)
        loss = 0.5 * linalg.norm(b - x.dot(A)) ** 2 + lmbda * linalg.norm(x, 1)
        losses.append(loss / n_samples)

        # if k % 500 == 0:
        #     print('Current loss:', loss)

    return x, losses


def update_dict(dictionary, Y, code, random_state=None, positive=False):
    """

    :param dictionary: array of shape (n_features, n_components)
        Value of the dictionary at the previous iteration.
    :param Y: array of shape (n_features, n_samples)
        Data matrix.
    :param code: array of shape (n_components, n_samples)
        Sparse coding of the data against which to optimize the dictionary.
    :param random_state:
    :param positive: boolean, optional
        Whether to enforce positivity when finding the dictionary.
    :return: dictionary : array of shape (n_components, n_features)
        Updated dictionary.
    """
    n_components = len(code)
    n_features = Y.shape[0]
    random_state = check_random_state(random_state)

    # Get BLAS functions
    gemm, = linalg.get_blas_funcs(('gemm',), (dictionary, code, Y))
    ger, = linalg.get_blas_funcs(('ger',), (dictionary, code))
    nrm2, = linalg.get_blas_funcs(('nrm2',), (dictionary,))

    # Residuals, computed with BLAS for speed and efficiency
    # R <- -1.0 * U * V^T + 1.0 * Y
    # Outputs R as Fortran array for efficiency
    R = gemm(-1.0, dictionary, code, 1.0, Y)

    for k in range(n_components):
        # R <- 1.0 * U_k * V_k^T + R
        R = ger(1.0, dictionary[:, k], code[k, :], a=R, overwrite_a=True)
        dictionary[:, k] = np.dot(R, code[k, :])
        if positive:
            np.clip(dictionary[:, k], 0, None, out=dictionary[:, k])
        # Scale k'th atom
        # (U_k * U_k) ** 0.5
        atom_norm = nrm2(dictionary[:, k])
        if atom_norm < 1e-10:
            dictionary[:, k] = random_state.randn(n_features)
            if positive:
                np.clip(dictionary[:, k], 0, None, out=dictionary[:, k])
            # Setting corresponding coefs to 0
            code[k, :] = 0.0
            # (U_k * U_k) ** 0.5
            atom_norm = nrm2(dictionary[:, k])
            dictionary[:, k] /= atom_norm
        else:
            dictionary[:, k] /= atom_norm
            # R <- -1.0 * U_k * V_k^T + R
            R = ger(-1.0, dictionary[:, k], code[k, :], a=R, overwrite_a=True)

    R = nrm2(R) ** 2.0
    return dictionary.T, R


def iterative_dict_learning_fista(shapes, n_components, dict_init=None, alpha=0.1,
                                  batch_size=100, n_iter=1000, random_state=None,
                                  if_shuffle=True, inner_stats=None, positive_dict=False):
    """Solves a dictionary learning matrix factorization problem online.
    Finds the best dictionary and the corresponding sparse code for
    approximating the data matrix X by solving::
        (U^*, V^*) = argmin 0.5 || X - U V ||_2^2 + alpha * || U ||_1
                     (U,V)
                     with || V_k ||_2 = 1 for all  0 <= k < n_components
    where V is the dictionary and U is the sparse code. This is
    accomplished by repeatedly iterating over mini-batches by slicing
    the input data.
    :param positive_dict:
    :param inner_stats : tuple of (A, B) ndarrays
        Inner sufficient statistics that are kept by the algorithm.
        Passing them at initialization is useful in online settings, to
        avoid losing the history of the evolution.
        A (n_components, n_components) is the dictionary covariance matrix.
        B (n_features, n_components) is the data approximation matrix
    :param if_shuffle:
    :param random_state:
    :param shapes: X, with shape n_samples, n_features
    :param n_components: n_atoms or n_basis
    :param dict_init:
    :param alpha: weight for the l1 term
    :param batch_size:
    :param n_iter:
    :param max_iter:
    :return: code (n_samples, n_components) and dictionary (n_component, n_feature)
    """

    n_samples, n_features = shapes.shape
    # Avoid integer division problems
    alpha = float(alpha)
    random_state = check_random_state(random_state)

    # Init V with SVD of X
    if dict_init is not None:
        dictionary = dict_init
    else:
        _, S, dictionary = randomized_svd(shapes, n_components, random_state=random_state)
        dictionary = S[:, np.newaxis] * dictionary
    r = len(dictionary)
    if n_components <= r:
        dictionary = dictionary[:n_components, :]
    else:
        dictionary = np.r_[dictionary,
                           np.zeros((n_components - r, dictionary.shape[1]))]

    if if_shuffle:
        X_train = shapes.copy()
        random_state.shuffle(X_train)
    else:
        X_train = shapes

    batches = gen_batches(n_samples, batch_size)
    batches = itertools.cycle(batches)

    # The covariance of the dictionary
    if inner_stats is None:
        A = np.zeros((n_components, n_components))
        # The data approximation
        B = np.zeros((n_features, n_components))
    else:
        A = inner_stats[0].copy()
        B = inner_stats[1].copy()

    losses = []
    for ii, batch in zip(range(n_iter), batches):
        this_X = X_train[batch]

        # calculate the sparse codes based on the current dict
        this_code, _ = fast_ista(this_X, dictionary, lmbda=alpha, max_iter=200)

        # Update the auxiliary variables
        if ii < batch_size - 1:
            theta = float((ii + 1) * batch_size)
        else:
            theta = float(batch_size ** 2 + ii + 1 - batch_size)
        beta = (theta + 1 - batch_size) / (theta + 1)

        A *= beta
        A += np.dot(this_code.T, this_code)
        B *= beta
        B += np.dot(this_X.T, this_code)

        dictionary, _ = update_dict(dictionary.T, B, A, random_state=random_state, positive=positive_dict)

        error = 0.5 * linalg.norm(np.matmul(this_code, dictionary) - this_X) ** 2 + alpha * linalg.norm(this_code, 1)
        error /= batch_size
        losses.append(error)

    # calucalte the codes for all images
    learned_codes, _ = fast_ista(shapes, dictionary, lmbda=alpha, max_iter=200)

    error = 0.5 * linalg.norm(np.matmul(learned_codes, dictionary) - shapes) ** 2 + alpha * linalg.norm(learned_codes,
                                                                                                        1)
    error /= n_samples
    print('Final Reconstruction error(frobenius norm): ', error)

    return dictionary, learned_codes, losses, error

