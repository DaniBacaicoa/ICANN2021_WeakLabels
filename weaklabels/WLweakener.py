#!/usr/bin/env python
# -*- coding: utf-8 -*-

# External modules
import numpy as np
from collections import Counter
# from numpy import binary_repr
from sklearn.preprocessing import label_binarize

# import sklearn.linear_model as sklm
import cvxpy
import copy


def weak_to_decimal(z):
    """
    >>> import numpy as np
    >>> z = np.array([[ 0.,  0.,  0.,  1.],
    ...               [ 0.,  0.,  1.,  0.],
    ...               [ 1.,  0.,  0.,  0.]])
    >>> weak_to_decimal(z)
    array([1, 2, 8])
    """
    n, n_cat = z.shape
    p2 = np.array([2**n for n in reversed(range(n_cat))])
    return np.array(z.dot(p2), dtype=int)


def weak_to_index(z, method='supervised'):
    """ Index position of weak labels in the corresponding mixing matrix

    It returns the row from the corresponding mixing matrix M where the weak
    label must be. For a supervised method the mixing matrix is a diagonal
    matrix withthe first row belonging to the first class and the last row
    belonging to the last class.

    With an Mproper, IPL, quasiIPL methods the mixing matrix is assumed to be
    2**#classes, where the first row corresponds to a weak labeling with all
    the labels to zero. The second row corresponds to the first class, and the
    last row corresponds to all the classes to one.


    >>> import numpy as np
    >>> z = np.array([[ 0.,  0.,  0.,  1.],
    ...               [ 0.,  0.,  1.,  0.],
    ...               [ 1.,  0.,  0.,  0.]])
    >>> weak_to_index(z, method='supervised')
    array([3, 2, 0])
    >>> weak_to_index(z, method='Mproper')
    array([1, 2, 8])
    >>> z = np.array([[ 0.,  0.,  0.,  0.],
    ...               [ 0.,  1.,  0.,  0.],
    ...               [ 1.,  0.,  1.,  1.]])
    >>> weak_to_index(z, method='Mproper')
    array([ 0,  4, 11])
    """

    # c = z.shape[1]
    if method in ['supervised', 'noisy', 'random_noise']:
        # FIXME which of both is correct?
        index = np.argmax(z, axis=1)
        # index = c - np.argmax(z, axis=1) - 1
    else:
        # index = np.array(map(bin_array_to_dec, z.astype(int)))
        index = weak_to_decimal(z)
    return index


def binarizeWeakLabels(z, c):
    """
    Binarizes the weak labels depending on the method used to generate the weak
    labels.

    Parameters
    ----------
    z : list of int
        List of weak labels. Each weak label is an integer whose binary
        representation encondes the observed weak labels
    c : int
        Number of classes. All components of z must be smaller than 2**c

    Returns
    -------
    z_bin : numpy.ndarray
        A 2-D array with c columns and as many rows as weak labels in z.
    """

    # Transform the weak label indices in z into binary label vectors
    z_bin = np.zeros((z.size, c), dtype=int)       # weak labels (binary)
    for index, i in enumerate(z):         # From dec to bin
        z_bin[index, :] = [int(x) for x in np.binary_repr(i, width=c)]

    return z_bin


def virtual_label_matrix(M, p=None, convex=True):
    """
    Computes the minimum MSE virtual label matrix for a given mixing matrix
    M and a given prior probability vector p, with or without a convexity
    restriction.

    Parameters
    ----------
    M : (n_weak_classes, c) numpy.ndarray
        Mixing matrix of floats corresponding to the stochastic
        process that generates the weak labels from the true labels.
    p : (n_weal_classes)-array-like or None
        Prior distribution of the weak classes. If None, a uniform distribution
        is assumed
    convex : boolean
        If true, compute the minimum MSE matrix, among those that can be used
        to build convex losses

    Returns
    -------
    Y : (n_samples, n_weak_labels) numpy.ndarray
        Virtual label matrix.

    Notes
    -----
    The case p=None and convex=False becomes equivalent to compute the Moore
    Penrose pseudo-inverse of M.
    In general, the case convex=False has an explicit formula, but involves a
    pseudoinverse computation. Maybe the cvx implementation is more efficient
    (though I did not test it)
    """

    d, c = M.shape
    if p is None:
        p = np.ones(d) / d

    # Identity matrix. I use Id instead of I to avoid pep-8 highlighting.
    Id = np.eye(c)

    c1 = np.ones((c, 1))
    d1 = np.ones((d, 1))
    hat_Y = cvxpy.Variable((c, d))

    if convex is True:
        # Compute the minimum mse convexity-preserving virtual matrix
        prob = cvxpy.Problem(
            cvxpy.Minimize(cvxpy.norm(
                cvxpy.hstack(
                    [cvxpy.norm(hat_Y[:, i])**2 * p[i] for i in range(d)]),
                1)),
            [hat_Y @ M == Id, hat_Y.T @ c1 == d1])
        prob.solve()
        Y = hat_Y.value

    else:
        # Compute the unconstrained minimum mse virtual matrix
        prob = cvxpy.Problem(
            cvxpy.Minimize(cvxpy.norm(
                cvxpy.hstack(
                    [cvxpy.norm(hat_Y[:, i])**2 * p[i] for i in range(d)]),
                1)),
            [hat_Y @ M == Id])
        prob.solve()
        Y = hat_Y.value

        # Ip = np.diag(1 / p)
        # Y = np.linalg.inv(M.T @ Ip @ M) @ M.T @ Ip

    return Y


def computeM(c, model_class, alpha=0.5, beta=0.5, gamma=0.5):
    """
    Generate a mixing matrix M, given the number of classes c.

    Parameters
    ----------
    alpha  : float, optional (default=0.5)
    beta   : float, optional (default=0.5)
    gamma  : float, optional (default=0.5)
    model_class : string, optional (default='supervised')
        Method to compute M. Available options are:
            'supervised': Identity matrix. For a fully labeled case
            'noisy': For a noisy label case: the true label is observed with
                probabiltity 1 - beta, otherwise one noisy label is taken at
                random
            'random_noise': All values of the mixing matrix are taken at random
                from a uniform distribution. The matrix is normalized to be
                left-stochastic
            'IPL':  Independent partial labels: the observed labels are
                independent. The true label is observed with probatility alpha.
                Each False label is observed with probability beta
            'IPL3': A generalized version of IPL, but only for c=3 classes and
                alpha=1: each false label is observed with a different
                probability. Parameters alpha, beta and gamma represent the
                probability of a false label for each column
            'quasi-IPL': The quasi independent partial label case discussed in
                the paper

    Returns
    -------
    M : array-like, shape = (n_classes, n_classes)
    """
    if model_class == 'supervised':

        M = np.eye(c)

    elif model_class == 'noisy':

        M = (np.eye(c) * (alpha - (1 - alpha) / (c - 1))
             + np.ones((c, c)) * (1 - alpha) / (c - 1))

    elif model_class == 'random_noise':

        M = np.random.rand(c, c)
        M = M / np.sum(M, axis=0, keepdims=True)

        M = (1 - beta) * np.eye(c) + beta * M

    elif model_class == 'random_weak':

        # Number or rows. Equal to 2**c to simulate a scenario where all
        # possible binary label vectors are possible.
        d = 2**c

        # Supervised component: Identity matrix with size d x c.
        Ic = np.zeros((d, c))
        for i in range(c):
            Ic[2**(c - i - 1), i] = 1

        # Weak component: Random weak label proabilities
        M = np.random.rand(d, c)
        M = M / np.sum(M, axis=0, keepdims=True)

        # Averaging supervised and weak components
        M = (1 - beta) * Ic + beta * M

    elif model_class == 'IPL':

        # Shape M
        d = 2**c
        M = np.zeros((d, c))

        # Compute mixing matrix row by row for the nonzero rows
        for z in range(0, d):

            # Convert the decimal value z to a binary list of length c
            z_bin = np.array([int(b) for b in bin(z)[2:].zfill(c)])
            modz = sum(z_bin)

            M[z, :] = (alpha**(z_bin) * (1 - alpha)**(1 - z_bin)
                       * (beta**(modz - z_bin)
                          * (1 - beta)**(c - modz - 1 + z_bin)))

        # This is likely not required: columns in M should already sum up
        # to 1
        M = M / np.sum(M, axis=0)

    elif model_class == 'IPL3':

        M = np.array([
            [0.0,                 0.0,               0.0],
            [0,                   0,                 (1 - gamma)**2],
            [0,                   (1 - beta)**2,     0],
            [0.0,                 beta * (1 - beta), gamma * (1 - gamma)],
            [(1 - alpha)**2,      0,                 0],
            [alpha * (1 - alpha), 0.0,               gamma * (1 - gamma)],
            [alpha * (1 - alpha), beta * (1 - beta), 0.0],
            [alpha**2,            beta**2,           gamma**2]])

    elif model_class == 'quasi-IPL':

        # Convert beta to numpy array
        if isinstance(beta, (list, tuple, np.ndarray)):
            # Make sure beta is a numpy array
            beta = np.array(beta)
        else:
            beta = np.array([beta] * c)

        # Shape M
        d = 2**c
        M = np.zeros((d, c))

        # Compute mixing matrix row by row for the nonzero rows
        for z in range(1, d - 1):

            # Convert the decimal value z to a binary list of length c
            z_bin = [int(b) for b in bin(z)[2:].zfill(c)]
            modz = sum(z_bin)

            M[z, :] = z_bin * (beta**(modz - 1) * (1 - beta)**(c - modz))

        # Remove zero rows
        M = M[1: d - 1, :]

        # Columns in M should sum up to 1
        M = M / np.sum(M, axis=0)

    else:
        raise ValueError("Unknown model to compute M: {}".format( model_class))

    return M


def generateM(c, model_class, alpha=0.2, beta=0.5):
    """
    Generate a mixing matrix M given some distribution parameters

    Parameters
    ----------
    alpha : float in [0, 1] or array-like (size=c), optional (default=0.2)
        Noise degree parameter. Higher values of this parameter usually
        mean higher label noise.
        The specific meaning of this parameter depends on the method:
        - 'supervised': Ignored.
        - 'noisy': noise probability (i.e. probability that the weak label
            does not correspond to the true label).
            If array-like, this probability is class-dependent
        - 'random_noise': Noise probability (same as 'noisy')
        - 'random_weak': Weak label probability. It is the probability that
            the weak label is generated at random.
            If array-like, this probability is class-dependent.
        - 'IPL': Missing label probability. It is the probability that the
            true label is not observed in the weak label.
            If array-like, this probability is class-dependent.
        - 'IPL3': Ignored
        - 'quasi-IPL': Ignored.

    beta : float (non-negative) or array-like, optional (default=0.5)
        Noise distribution parameter.
        The specific meaning of this parameter depends on the method:
        - 'supervised': Ignored.
        - 'noisy': Ignored
        - 'random_noise': Concentration parameter. The noisy label
            probabilities are generated stochastically according to a
            Dirichlet distribution with parameters beta. According to this:
                - beta = 1 is equivalent to a uniform distribution
                - beta = inf is equivalent to using option 'noisy': the
                    class of the noisy label is random.
                - beta < 1 implies higher concentration: most noise
                    probability gets concentrated in a single class. This
                    may be useful to simulate situations where a class is
                    usually mixed with another similar clas, but not with
                    others.
            If beta is array-like, a different concentration parameter will
            be used for each class (i.e. for each column of M)
        - 'random_weak': Concentration parameter of the weak label
            probability distribution, which is a Dirichlet.
                - beta = 1 is equivalent to a uniform distribution
                - beta = inf is equivalent to a constant probability over
                    all weak labels
                - beta < 1 implies higher concentration: most probability
                    mass is concentrated over a few weak labels
            If beta is array-like, a different concentration parameter will
            be used for each class (i.e. for each column of M)
        - 'IPL': Probability that a noisy label from a given class is
            observed. If array-like, this probability is class-dependent:
            beta[c] is the probability that, if the true label is not c,
            the weak label contains c
        - 'IPL3': Probability that a noisy label from any class is
            observed. If array-like, this probability is class-dependent:
            beta[c] is the probability that, if the true label is c, the
            weak label contains a label from class c' other than c
        - 'quasi-IPL': .

    Returns
    -------
    M : array-like, shape = (n_weak_classes, c)
    """

    # Change infinite for a very large number
    beta = np.nan_to_num(beta)
    if model_class == 'supervised':

        M = np.eye(c)

    elif model_class == 'noisy':

        valpha = np.array(alpha)
        M = (np.eye(c) * (1 - valpha - valpha / (c - 1))
             + np.ones((c, c)) * valpha / (c - 1))

    elif model_class == 'random_noise':
        # Diagonal component (no-noise probabilities)
        # np.array is used just in case beta is a list
        D = (1 - np.array(alpha)) * np.eye(c)

        # Non-diagonal components
        # Transforma beta into an np.array (if it isn't it).
        vbeta = np.array(beta) * np.ones(c)
        B = np.random.dirichlet(vbeta, c).T

        # Remove diagonal component and rescale
        # I am using here the fact that the conditional distribution of a
        # rescaled subvector of a dirichlet is a dirichet with the same
        # parameters, see
        # https://math.stackexchange.com/questions/1976544/conditional-
        # distribution-of-subvector-of-a-dirichlet-random-variable
        # Conditioning...
        B = B * (1 - np.eye(c))
        # Rescaling...
        B = B / np.sum(B, axis=0)
        # Rescale by (1-beta), which are the probs of noisy labels
        B = B @ (np.eye(c) - D)

        # Compute M
        M = D + B

    elif model_class == 'random_weak':
        # Number or rows. Equal to 2**c to simulate a scenario where all
        # possible binary label vectors are possible.
        d = 2**c

        # Supervised component: Identity matrix with size d x c.
        Ic = np.zeros((d, c))
        for i in range(c):
            Ic[2**(c - i - 1), i] = 1

        # Weak component: Random weak label proabilities
        # Transforma beta into an np.array (if it isn't it).
        vbeta = np.array(beta) * np.ones(d)
        B = np.random.dirichlet(vbeta, c).T

        # Averaging supervised and weak components
        # np.array is used just in case alpha is a list
        M = (1 - np.array(alpha)) * Ic + np.array(alpha) * B

    elif model_class == 'IPL':

        # Shape M
        d = 2**c
        M = np.zeros((d, c))

        valpha = np.array(alpha)
        vbeta = np.array(beta)

        # Compute mixing matrix row by row for the nonzero rows
        for z in range(0, d):

            # Convert the decimal value z to a binary list of length c
            z_bin = np.array([int(b) for b in bin(z)[2:].zfill(c)])
            modz = sum(z_bin)

            M[z, :] = (((1 - valpha) / vbeta)**z_bin
                       * (valpha / (1 - vbeta))**(1 - z_bin)
                       * np.prod(vbeta**z_bin)
                       * np.prod((1 - vbeta)**(1 - z_bin)))

    elif model_class == 'IPL3':

        b0 = beta[0]
        b1 = beta[1]
        b2 = beta[2]

        M = np.array([
            [0.0, 0.0, 0.0],
            [0, 0, (1 - b2)**2],
            [0, (1 - b1)**2, 0],
            [0.0, b1 * (1 - b1), b2 * (1 - b2)],
            [(1 - b0)**2, 0, 0],
            [b0 * (1 - b0), 0.0, b2 * (1 - b2)],
            [b0 * (1 - b0), b1 * (1 - b1), 0.0],
            [b0**2, b1**2, b2**2]])

    elif model_class == 'quasi-IPL':

        beta = np.array(beta)

        # Shape M
        d = 2**c
        M = np.zeros((d, c))

        # Compute mixing matrix row by row for the nonzero rows
        for z in range(1, d - 1):

            # Convert the decimal value z to a binary list of length c
            z_bin = np.array([int(b) for b in bin(z)[2:].zfill(c)])
            modz = sum(z_bin)

            M[z, :] = z_bin * (beta**(modz - 1) * (1 - beta)**(c - modz))

        # Columns in M should sum up to 1
        M = M / np.sum(M, axis=0)

    else:
        raise ValueError(f"Unknown model to compute M: {model_class}")

    return M


class WLmodel(object):
    """
    A class to model a weak labelling process. Includes methods to define a
    model (through different types of reconstruction matrices), to generate
    weak labels from a given set of clean labels, and to compute virtual
    labels using different methods
    """

    def __init__(self, c, model_class=None, weak_classes=None):
        """
        Initializes a Weak Label model

        Parameters
        ----------
        c : int
            Number of classes
        model_class : str or None
            Type of model. Default falue is None, which means that no model is
            specified. In that case, weak_classes must be not None.

            Different models are defined by different types of mixing matrices.
            Available models are:
            - 'supervised': Identity matrix. For a fully labeled case.
            - 'noisy': For a noisy label case with deterministic parameters:
                    The true label is observed with a given probability,
                    otherwise one noisy label is taken at random. Parameter
                    alpha is deterministic.
            - 'random_noise': Noisy labels with stochastic parameters.
                    Same as 'noixy', but the parameters of the noise
                    distribution are generated at random.
            - 'random_weak': A generic mixing label matrix with stochastic
                    components
            - 'IPL': Independent partial labels: the observed labels are
                    independent. The true label is observed with probability
                    alpha. Each False label is observed with probability beta.
            - 'IPL3': It is a generalized version of IPL, but only for c=3
                    classes and alpha=1: each false label is observed with a
                    different probability. Parameters alpha, beta and gamma
                    represent the probability of a false label for each column.
            - 'quasi-IPL': This is the quasi-independent partial label case:
                    the probability of any weak label depends on the number of
                    false labels only.

        weak_classes : list or str {'one-hot', 'non-uniform', 'all'}
            Defines all possible weak classes. Each weak class will be
            represented by an integer whose binary representation encodes
            the classes that contains.
            E.g. assume c=4 (classes 0,1,2 and 3). The weak class containing
            classes 0 and 2 is given by the binary code '0011' (ones in
            positions 0 and 2). Thus, the weak class is 3
            - If list, the possible weak classes are given explicitely.
            - If 'one-hot': each class contains only one class. For c=4, weak
                classes are '0001', '0010', '0100' and '1000', i.e.: 1, 2, 4
                and 8
            - If 'mixed': each weak class contains at least one class but not
                all of them, i.e. cases '00...0' and '11...1' are excluded
            - If 'all': the weak class migh containg an abitrary number of
                classes.
        """

        # One of model_class or weak labels  must be provided, but not both
        # This is checked here
        if model_class is None and weak_classes is None:
            raise ValueError(
                "One of the parameters 'model_class' or 'weak_classes' should"
                "be given")
        if model_class is not None and weak_classes is not None:
            raise ValueError(
                "You cannot specify both parameters 'model_class' and"
                "'weak_classes'")

        self.c = c
        self.model_class = model_class
        self.M = None

        # State list of weak classes
        if weak_classes is None:
            # The list of weak_classes will be inferred from the model_class
            if model_class in ['supervised', 'noisy', 'random_noise']:
                self.weak_classes = 2**np.arange(c - 1, -1, -1)
            elif model_class in ['random_weak', 'IPL', 'IPL3', 'quasi-IPL']:
                self.weak_classes = np.arange(2**c)
            # I introduce this (quasi-IPL' in the one above to be consistent
            # when applying remove_zero_rows
            # elif model_class in ['quasi-IPL']:
            #    self.weak_classes = np.arange(1, 2**c - 1)
            else:
                raise ValueError("Unknown model_class: {}".format(model_class))
        elif weak_classes == 'one-hot':
            self.weak_classes = 2**np.arange(c - 1, -1, -1)
        elif weak_classes == 'non-uniform':
            self.weak_classes = np.arange(1, 2**c - 1)
        elif weak_classes == 'all':
            self.weak_classes = np.arange(2**c)
        else:
            # The list of weak classes is given explicitely
            self.weak_classes == weak_classes

        return

    def generateM(self, alpha=0.2, beta=0.5):
        M = generateM(c=self.c, model_class=self.model_class, alpha=alpha,
                      beta=beta)
        self.M = copy.copy(M)
        return M

    def loadM(self, M):
        """
        Load a mixing matrix in the weak label model.

        Parameters
        ----------
        M: numpy.ndarray
            Mixing matrix: a column-stochastic matrix with self.c columns and
            as many rows as the size of self.weak_classes
        """

        nc, nw = M.shape

        if nc != self.c or nw != len(self.weak_classes):
            raise ValueError(
                f"The size of the mixing matrix is wrong. It must have "
                f"{self.c} columns and {len(self.weak_classes)} rows")

        self.M = copy.copy(M)

        return

    def remove_zero_rows(self):
        """
        Removes the weak classes with zero probability, i.e. those whose
        corresponding row in M is all zeros.
        """

        # Find nonzero rows
        flag = np.nonzero(np.sum(self.M, axis=1))[0]

        # Remove zero rows from self.M and self.weak_classes
        self.M = self.M[flag, :]
        self.weak_classes = self.weak_classes[flag]

        return

    def computeM(self, alpha=0.5, beta=0.5, gamma=0.5):
        """
        Generate a mixing matrix M for the given parameters.

        Parameters
        ----------
        alpha  : float, optional (default=0.5)
        beta   : float, optional (default=0.5)
        gamma  : float, optional (default=0.5)

        Notes
        -----
        The meaning of the input parameters depend on the model class 
        (in self.model_class)

        Returns
        -------
        M : array-like, shape = (n_classes, n_classes)
        """
        M = computeM(c=self.c, model_class=self.model_class, alpha=alpha,
                     beta=beta, gamma=gamma)
        self.M = copy.copy(M)
        return M

    def generateWeak(self, y):
        """
        Generate the set of weak labels z from the ground truth labels y, given
        a mixing matrix M and, optionally, a set of possible weak labels, zset.

        Parameters
        ----------
        y : list
            List of true labels with values from 0 to c-1 where c is the
            number of classes

        Returns
        -------
        z : list
            List of weak labels. Each weak label is an integer whose binary
            representation encodes the observed weak labels.
        """

        z = np.zeros(y.shape, dtype=int)  # Weak labels for all labels y (int)

        # weak_classes = np.arange(d)    # Possible weak labels (int)
        for index, i in enumerate(y):
            z[index] = np.random.choice(self.weak_classes, 1, p=self.M[:, i])

        return z

    def estimate_wl_priors(self, z, loss='cross_entropy'):
        """
        Estimate the prior weak label probabilities based on the samples in z.
        The estimates are computed as the projection of the standard frequency-
        based estimate onto the simplex generated by the mixing matrix.

        The estimation is stated as an optimization problem. If n is the
        vector of counts of the weak labels in z, the co

            p_reg = arg min_p l(n, p), 
            s.t.  p = M eta, and eta in Pc, where Pc is the simplex.

        l(n, p) is the loss function.

        Parameters
        ----------
        z : array-like of int
            Observed weak labels. Each weak label must be one of the integers
            in self.weak_classes

        loss : str, optional (default='cross_entropy')
            Optimization criteria used to compute the projection. Available
            options are:
                'cross_entropy': l(n, p) = n.T log(p)
                'square_error': l(n, p) = ||n/sum(n) - p||**2

        Returns
        -------
        p_reg : numpy.ndarray
            Prior weak label probabilities
        """

        z_count = Counter(z)
        # Make sure all values in z are in self.weak_classes
        test = [x for x in z_count if x not in self.weak_classes]
        if len(test) > 0:
            print("WARNING: values {x} are not weak classes")

        p_est = np.array([z_count[x] for x in self.weak_classes])
        # This is commented, because normalization is actually not necessary.
        # p_est = p_est / np.sum(p_est)

        # Resolution of the problem in eq (32) and (33)
        v_eta = cvxpy.Variable(self.c)
        if loss == 'cross_entropy':
            lossf = -p_est @ cvxpy.log(self.M @ v_eta) 
        elif loss == 'square_error':
            p_est = p_est / np.sum(p_est)
            lossf = cvxpy.sum_squares(p_est - self.M @ v_eta)

            # For the square error, cvxpy is likely not necessary. I think
            # the estimate can be computed as
            # p_reg = self.M @ np.linalg.lstsq(self.M, p_est, rcond=None)[0]
            # p_reg = (p_reg > 0) * p_reg
            # p_reg = p_reg / np.sum(p_reg)

        else:
            exit("ERROR: unknown loss")

        # State and solve the optimization problem over eta
        problem = cvxpy.Problem(cvxpy.Minimize(lossf),
                                [v_eta >= 0, np.ones(self.c) @ v_eta == 1])
        problem.solve()

        # Compute the wl prior estimate
        p_reg = self.M @ v_eta.value

        return p_reg   # , v_eta.value

    def virtual_labels(self, z, method, p=None):
        """
        Generate the set of virtual labels v for the (decimal) weak labels in
        z, given a weak label model in variable method and, optionally, a
        mixing matrix M, and a list of admissible decimal labels.

        Warning: This method is deprecated. Use compute_virtual_labels instead

        Parameters
        ----------
        z : list
            Weak labels. Each weak label is an integer whose binary
            representation encodes the observed weak labels

        method : str
            Method for computation of the virtual label vector v
            Available methods are:
                'binary': Takes virtual labels equal to the binary
                    representations of the weak labels in z
                'quasi-IPL': Generic virtual labels based on a 'quasi-IPL'
                    mixing matrix
                'M-pinv': Virtual labels based on the Moore-Penrose
                    pseudo-inverse of M
                'M-conv': Virtual labels based on a left inverse of M that
                    guarantees convexity of the appropriate weak loss
                'M-opt':  Virtual labels minimizing MSE.
                'M-opt-conv': Convexity-preserving virtual labels minimizing
                    MSE

        p : numpy.ndarray or None, optional (default=None)
            Weak label priors. If None, they are estimated fron z.
            Only for methods 'M-opt' and 'M-opt-conv'

        Returns
        -------
        v : numpy.ndarray
            Virtual labels
        """

        if method == 'binary':
            # The virtual labels are taken as binarized versions of the weak
            # labels
            v = binarizeWeakLabels(z, self.c).astype(float)
        elif method == 'quasi-IPL':    # quasi-independent labels
            # The virtual labels are computed from the weak label vectors
            v = binarizeWeakLabels(z, self.c).astype(float)

            # Each 1 or 0 in the weak label vector must be replaced by a number
            # that depends on the total number of 1's in the vector
            for index in range(len(v)):
                aux = v[index, :]
                weak_sum = np.sum(aux)
                if weak_sum != self.c:
                    weak_zero = float(1 - weak_sum) / (self.c - weak_sum)
                    aux[aux == 0] = weak_zero
                    v[index, :] = aux
                else:
                    # In the quasi-IPL method, it is assumed that nor z=0 nor
                    # z=2**C will happen. A zero vector is assigned here, just
                    # in case, though the choice is arbitrary.
                    # TODO MPN I changed Nans to zeros. Is this important?
                    v[index, :] = np.array([None] * self.c)

        elif method == 'M-pinv':
            # (This could be computed as ...
            # v = self.virtual_labels_from_M(z, M, p, optimize=False,
            #                                convex=False)
            # ... but the following is equivalent.
            Y = np.linalg.pinv(self.M)
            # Compute the virtual labels
            v = self.virtual_labels_from_Y(z, Y)

        elif method == 'M-conv':
            # Compute the virtual label matrix
            v = self.virtual_labels_from_M(z, optimize=False, convex=True)
        elif method == 'M-opt':
            # Compute the virtual label matrix
            v = self.virtual_labels_from_M(
                z, p=p, optimize=True, convex=False)
        elif method == 'M-opt-conv':
            # Compute the virtual label matrix
            v = self.virtual_labels_from_M(
                z, p=p, optimize=True, convex=True)
        else:
            raise ValueError(
                f"Unknown method {method} to create virtual labels")

        return v

    def virtual_labels_from_M(self, z, p=None, optimize=True, convex=True):
        """
        Generate the set of virtual labels v for the (decimal) weak labels in
        z, given a mixing matrix M.

        The virtual label matrix is computed in order to minimize the MMSE for
        a given prior p.

        If p is unknown, it is estimated from z (if optimized=True) or taken
        as uniform (if optimize=False)

        Parameters
        ----------
        z : list
            Weak labels. Each weak label is an integer whose binary
            representation encodes the observed weak labels

        p : numpy.ndarray or None, optional (default=None)
            Weak label priors. If None, they are estimated fron z or taken as
            uniform, depending on the value of optimized.
                If optimized=True, p will be estimated from z
                If optimized=False, p will be taken as a uniform distribution

        optimize : boolean, optional (default=True)
            If True, the virtual label matrix is computed for the given value
            of p or for an estimate based on z.
            If False, the given p is ignored, and a uniform distribution is
            used

        convex : boolean, optional (default=True)
            If True, the virtual label matrix is selected in order to guarantee
            that the resulting weak loss can be convex. If False, no convexity
            constraints are imposed.

        Returns
        -------
        v : numpy.ndarray
            Virtual labels
        """

        if optimize:
            if p is None:
                # Estimate the prior weak label probs from from data
                p = self.estimate_wl_priors(z)

        else:
            # Note that in this case the given value of p is ignored
            p = None

        # Compute the virtual label matrix ...
        Y = virtual_label_matrix(self.M, p, convex=convex)
        # ... and use it to ompute the virtual labels
        v = self.virtual_labels_from_Y(z, Y)

        return v

    def virtual_labels_from_Y(self, z, Y):
        """
        Computes the virtual labels corresponding to the weak labels in z for a
        virtual label matrix Y.

        Parameters
        ----------
        z : array-like of int
            Weak labels (in integer format)
        Y : numpy.ndarray
            Virtual label matrix
        """

        # Compute inverted index from decimal labels to position in
        # self.weak_classes
        z2i = dict(list(zip(self.weak_classes,
                            list(range(len(self.weak_classes))))))

        # Compute the virtual label.
        v = np.zeros((z.size, self.c))

        for i, zi in enumerate(z):
            # The virtual label for the i-th weak label, zi, is the column
            # in Y corresponding to zi (that is taken from the inverted index)
            v[i, :] = Y[:, z2i[zi]]

        return v

