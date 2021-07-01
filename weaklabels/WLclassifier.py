#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Defines classifier objects that work with weak labels

    Author: JCS, May. 2016
"""

import numpy as np
import scipy as sp


class WeakLogisticRegression(object):
    """
    A classifier object for multiclass logistic regression from weak labels.
    """

    def __init__(self, n_classes=2, method="VLL", optimizer='GD',
                 canonical_link=False, params={}, sound='off'):
        """
        Initializes a classifier object

        Parameters
        ----------
        n_classes : int, optional (default=2)
            Number of classes
        method : str, optional (default=VLL)
            Available options are
            - VLL: Learning based on virtual labels
            - OSL: Optimistic Superset Learning (a decision-directed method)
            - EM: Expectation maximization
        optimizer : str, optional (default='GD')
            Optimization method. If 'GD', a self implementation of stochastic
            gradient descent is used. Otherwise, a method from scipy.optimize
            is used.
        canonical_link : booblean, optinal (default=False)
            If True, the probabilistic predictions use the canonical link.
            Otherwise, the logistic function is used.
        params : dict, optinal (default={})
            Parameters for the optimization algorithms.
            Available parameters are:
            - n_it: Number of iterations
            - rho: learning step
            - alpha: regularization constant
            - loss: loss function (CE: cross entropy, square: square error)
            -
        """

        self.sound = sound
        self.params = params
        self.method = method
        self.optimizer = optimizer
        self.canonical_link = canonical_link
        self.n_classes = n_classes
        self.classes_ = list(range(n_classes))

        # Set default value of parameter 'alpha' if it does not exist
        if self.method == "VLL" and 'alpha' not in self.params:
            self.params['alpha'] = 0

        # This is for backwards compatibility
        if 'loss' not in self.params:
            self.params['loss'] = 'CE'

    def softmax(self, x):
        """
        Computes the softmax transformation

        Args:
            :x  : NxC matrix of N samples with dimension C

        Returns:
            :p  : NxC matrix of N probability vectors with dimension C
        """

        # Shift each x to have maximum value=1. This is to avoid exponentiating
        # large numbers that cause overflow
        z = x - np.max(x, axis=1, keepdims=True)
        p = np.exp(z)
        p = p / np.sum(p, axis=1, keepdims=True)

        return p

    def logsoftmax(self, x):
        """
        Computes the elementwise logarithm of the softmax transformation

        Args:
            :x  : NxC matrix of N samples with dimension C

        Returns:
            :p  : NxC matrix of N probability vectors with dimension C
        """

        # Shift each x to have maximum value=1. This is to avoid exponentiating
        # large numbers that cause overflow
        z = x - np.max(x, axis=1, keepdims=True)
        logp = z - np.log(np.sum(np.exp(z), axis=1, keepdims=True))

        return logp

    def index2bin(self, vector, dim):
        """
        Converts an array of indices into a matrix of binary vectors

        Adapted from "http://stackoverflow.com/questions/23300715/
                      numpy-transform-vector-to-binary-matrix"
        (Check the web link to see a faster sparse version that is much more
        efficient for large dimensions)

        Args:
           :vector: Array of integer indices 0, 1, ..., dim-1
           :dim: Dimension of the output vector.
        """

        n = vector.shape[0]
        v_bin = np.zeros((n, dim))
        v_bin[np.arange(n), vector] = 1

        return v_bin

    def hardmax(self, Z):
        """
        Transform each row in array Z into another row with zeroes in the
        non-maximum values and 1/nmax on the maximum values, where nmax is the
        number of elements taking the maximum value
        """

        D = sp.equal(Z, np.max(Z, axis=1, keepdims=True))

        # In case more than one value is equal to the maximum, the output
        # of hardmax is nonzero for all of them, but normalized
        D = D / np.sum(D, axis=1, keepdims=True)

        return D

    def squareLoss(self, w, X, T):
        """
        Compute a regularized square loss (a.k.a Brier score) for samples in X,
        virtual labels in T and parameters w. The regularization parameter is
        taken from the object attributes.
        It assumes a multi-class softmax classifier.
        This method implements two different log-losses, that are specified in
        the object's attribute self.method:
            'OSL' :Optimistic Superset Loss. It assumes that the true label is
                   the nonzero weak label with highest posterior probability
                   given the model.
            'VLL' :Virtual Labels Loss.
        The regularization parameter is set in set.params['alpha']

        Args:
            :w:  1-D nympy array. A flattened version of the weight matrix of
                 the multiclass softmax. This 1-D arrangement is required by
                 the scipy optimizer that will use this method.
            :X:  Input data. An (NxD) matrix of N samples with dimension D
            :T:  Target class. An (NxC) array of target vectors.
                 The meaning and format of each target vector t depends
                 on the selected log-loss version:
                 - 'OSL': t is a binary vector.
                 - 'VLL': t is a virtual label vector
        Returns:
            :L:  Log-loss
        """

        n_dim = X.shape[1]
        W2 = w.reshape((n_dim, self.n_classes))
        logp = self.logsoftmax(np.dot(X, W2))
        p = np.exp(logp)

        if self.method == 'OSL':
            D = self.hardmax(T * p)
            L = np.sum((D - p)**2) / 2

        else:
            # Compute the square loss for virtual label vector T. I am
            # assuming that there is no bias term is the square losss. Maybe
            # I should verify it, though it do not this it would be relevant.
            L = np.sum((T - p)**2) / 2

        return L

    def logLoss(self, w, X, T):
        """
        Compute a (possibly regularized) log loss (cross-entropy) for samples
        in X, virtual labels in T and parameters w. The regularization
        parameter is taken from attribute self.params['alpha']
        It assumes a multi-class softmax classifier.
        This method implements different log-losses, that are specified in
        the object's attribute self.method:
            'OSL' :Optimistic Superset Loss. It assumes that the true label is
                   the nonzero weak label with highest posterior probability
                   given the model.
            'VLL' :Virtual Labels Loss.
            'EM'  :Expected Log likelihood (i.e. the expected value of the
                   complete data log-likelihood after the E-step). It assumes
                   that a mixing matrix is known and contained in
                   self.params['M']

        Args:
            :w:  1-D nympy array. A flattened version of the weight matrix of
                 the multiclass softmax. This 1-D arrangement is required by
                 the scipy optimizer that will use this method.
            :X:  Input data. An (NxD) matrix of N samples with dimension D
            :T:  Target class. An (NxC) array of target vectors.
                 The meaning and format of each target vector t depends
                 on the selected log-loss version:
                 - 'OSL': t is a binary vector.
                 - 'VLL': t is a virtual label vector
                 - 'EM': t is an integer index of a weak label
        Returns:
            :L:  Log-loss
        """

        n_dim = X.shape[1]
        W2 = w.reshape((n_dim, self.n_classes))
        logp = self.logsoftmax(np.dot(X, W2))

        if self.method == 'OSL':
            p = np.exp(logp)
            D = self.hardmax(T * p)
            L = -np.sum(D * logp)

        elif self.method == 'EM':
            M = self.params['M']
            p = np.exp(logp)
            Q = p * M[T, :].T
            Q = Q / np.sum(Q, axis=0)
            L = -np.sum(Q * logp)

        else:
            L = - np.sum(T * logp) + self.params['alpha'] * np.sum(w**2) / 2

        # if L < 0:
        #    warnings.warn(f"Negative log-loss (L={L}): use larger alpha)")

        return L

    def LBLloss(self, w, X, T):
        """
        Compute a (possibly regularized) log loss (cross-entropy) for samples
        in X, virtual labels in T and parameters w. The regularization
        parameter is taken from attribute self.params['alpha']
        It assumes a multi-class softmax classifier.
        This method implements different log-losses, that are specified in
        the object's attribute self.method:
            'OSL' :Optimistic Superset Loss. It assumes that the true label is
                   the nonzero weak label with highest posterior probability
                   given the model.
            'VLL' :Virtual Labels Loss.
            'EM'  :Expected Log likelihood (i.e. the expected value of the
                   complete data log-likelihood after the E-step). It assumes
                   that a mixing matrix is known and contained in
                   self.params['M']

        Args:
            :w:  1-D nympy array. A flattened version of the weight matrix of
                 the multiclass softmax. This 1-D arrangement is required by
                 the scipy optimizer that will use this method.
            :X:  Input data. An (NxD) matrix of N samples with dimension D
            :T:  Target class. An (NxC) array of target vectors.
                 The meaning and format of each target vector t depends
                 on the selected log-loss version:
                 - 'OSL': t is a binary vector.
                 - 'VLL': t is a virtual label vector
                 - 'EM': t is an integer index of a weak label
        Returns:
            :L:  Log-loss
        """

        k2 = self.params['k'] / 2
        alpha = self.params['alpha']
        beta = self.params['beta']

        n_dim = X.shape[1]
        W2 = w.reshape((n_dim, self.n_classes))
        V = X @ W2
        # Forze each v_i to lie in the orthogonal subspace:
        # V = X W - (X·W·1) 1'   (where 1 is a c-ones vector)
        V -= np.mean(V, axis=1, keepdims=True)

        logp = self.logsoftmax(V)

        if self.method == 'OSL':
            p = np.exp(logp)
            D = self.hardmax(T * p)
            L = -np.sum(D * logp)

        else:
            L = (- np.sum(T * logp) + alpha * np.sum(w**2) / 2
                 + k2 * np.sum(np.abs(V)**beta))

        # if L < 0:
        #     warnings.warn(f"Negative log-loss (L={L}): use larger alpha)")

        return L

    def loss(self, w, X, T):

        if self.params['loss'] == 'CE':

            L = self.logLoss(w, X, T)

        elif self.params['loss'] == 'LBL':

            L = self.LBLloss(w, X, T)

        elif self.params['loss'] == 'square':

            L = self.squareLoss(w, X, T)

        else:

            exit('Unknown loss')

        return L

    def gradSquareLoss(self, w, X, T):
        """
        Compute the gradient of the square loss (Brier score) for
        samples in X, virtual labels in T and parameters w.
        It assumes a multi-class softmax classifier.
        This method implements gradients for two different square losses, that
        are specified in the object's attribute self.method:
            'OSL' :Optimistic Superset Loss. It assumes that the true label is
                   the nonzero weak label with highest posterior probability
                   given the model.
            'VLL' :Virtual Labels Loss.

        Args:
            :w:  1-D nympy array. A flattened version of the weight matrix of
                 the multiclass softmax. This 1-D arrangement is required by
                 the scipy optimizer that will use this method.
                :X:  Input data. An (NxD) matrix of N samples with dimension D
                :T:  Target class. An (NxC) array of target vectors.
                     The meaning and format of each target vector t depends on
                     the selected log-loss version:
                     - 'OSL': t is a binary vector.
                     - 'VLL': t is a virtual label vector
            Returns:
                :G:  Gradient of the Log-loss
        """

        n_dim = X.shape[1]
        W2 = w.reshape((n_dim, self.n_classes))
        p = self.softmax(np.dot(X, W2))

        if self.method == 'OSL':
            D = self.hardmax(T * p)
            Q = (p - D) * p
        else:
            Q = (p - T) * p

        sumQ = np.sum(Q, axis=1, keepdims=True)
        G = np.dot(X.T, Q - sumQ * p)

        return G.reshape((n_dim * self.n_classes))

    def gradLogLoss(self, w, X, T):
        """
        Compute the gradient of the regularized log loss (cross-entropy) for
        samples in X, virtual labels in T and parameters w.
        The regularization parameter is taken from the object attribute
        self.params['alpha']
        It assumes a multi-class softmax classifier.
        This method implements gradients for two different log-losses, that are
        specified in the object's attribute self.method:
            'OSL' :Optimistic Superset Loss. It assumes that the true label is
                   the nonzero weak label with highest posterior probability
                   given the model.
            'VLL' :Virtual Labels Loss.
            'EM'  :Expected Log likelihood (i.e. the expected value of the
                   complete data log-likelihood after the E-step). It assumes
                   that a mixing matrix is known and contained in
                   self.params['M']

        Parameters
        ----------
        w: 1-D nympy array
            A flattened version of the weight matrix of the multiclass softmax.
            This 1-D arrangement is required by the scipy optimizer that will
            use this method.
        X:  numpy.ndarray (N x D)
            Input data. An (NxD) matrix of N samples with dimension D
        T:  Target class. An (NxC) array of target vectors.
            The meaning and format of each target vector t depends on the
            selected log-loss version:
            - 'OSL': t is a binary vector.
            - 'VLL': t is a virtual label vector
            - 'EM': t is an integer index of a weak label

        Returns
        -------
        G:  Gradient of the Log-loss
        """

        n_dim = X.shape[1]
        W2 = w.reshape((n_dim, self.n_classes))
        p = self.softmax(np.dot(X, W2))

        if self.method == 'OSL':
            D = self.hardmax(T * p)
            G = np.dot(X.T, p - D)

        elif self.method == 'EM':
            M = self.params['M']
            Q = p * M[T, :].T
            Q = Q / np.sum(Q, axis=0)
            G = np.dot(X.T, p - Q)

        else:
            sumT = np.sum(T, axis=1, keepdims=True)
            G = np.dot(X.T, p * sumT - T) - self.params['alpha'] * W2
            # G = np.dot(X.T, p - T)

        return G.reshape((n_dim * self.n_classes))

    def gradLBL(self, w, X, T):
        """
        Compute the gradient of the regularized log loss (cross-entropy) for
        samples in X, virtual labels in T and parameters w.
        The regularization parameter is taken from the object attribute
        self.params['alpha']
        It assumes a multi-class softmax classifier.
        This method implements gradients for two different log-losses, that are
        specified in the object's attribute self.method:
            'OSL' :Optimistic Superset Loss. It assumes that the true label is
                   the nonzero weak label with highest posterior probability
                   given the model.
            'VLL' :Virtual Labels Loss.
            'EM'  :Expected Log likelihood (i.e. the expected value of the
                   complete data log-likelihood after the E-step). It assumes
                   that a mixing matrix is known and contained in
                   self.params['M']

        Parameters
        ----------
        w: 1-D nympy array
            A flattened version of the weight matrix of the multiclass softmax.
            This 1-D arrangement is required by the scipy optimizer that will
            use this method.
        X:  numpy.ndarray (N x D)
            Input data. An (NxD) matrix of N samples with dimension D
        T:  Target class. An (NxC) array of target vectors.
            The meaning and format of each target vector t depends on the
            selected log-loss version:
            - 'OSL': t is a binary vector.
            - 'VLL': t is a virtual label vector
            - 'EM': t is an integer index of a weak label

        Returns
        -------
        G:  Gradient of the Log-loss
        """

        k2 = self.params['k'] / 2
        alpha = self.params['alpha']
        beta = self.params['beta']

        n_dim = X.shape[1]
        W2 = w.reshape((n_dim, self.n_classes))
        V = X @ W2
        # Forze each row in V to lie in the orthogonal subspace:
        # V = X W - (X·W·1) 1'   (where 1 is a c-ones vector)
        V -= np.mean(V, axis=1, keepdims=True)

        p = self.softmax(V)

        # Correction term
        delta = np.abs(V)**(beta - 1) * np.sign(V)
        delta -= np.sum(delta, axis=1, keepdims=True) / self.n_classes
        delta = k2 * beta * delta

        if self.method == 'OSL':
            D = self.hardmax(T * p)
            G = X.T @ (p - D)

        else:
            sumT = np.sum(T, axis=1, keepdims=True)
            G = X.T @ (p * sumT - T) - alpha * W2
            G += X.T @ delta

        return G.reshape((n_dim * self.n_classes))

    def gradLoss(self, w, X, T):

        if self.params['loss'] == 'CE':
            # Cross entropy (log-loss)
            g = self.gradLogLoss(w, X, T)

        elif self.params['loss'] == 'square':
            # Square error
            g = self.gradSquareLoss(w, X, T)

        elif self.params['loss'] == 'LBL':
            # Lower-Bounded Log-loss.
            g = self.gradLBL(w, X, T)

        else:

            exit('Unknown loss')

        return g

    def gd(self, X, T):
        """
        Trains a logistic regression classifier by a gradient descent method
        """

        # Initialize variables
        n_dim = X.shape[1]
        W = np.random.randn(n_dim, self.n_classes)
        w1 = W.reshape((n_dim * self.n_classes))

        # Running the gradient descent algorithm
        for n in range(self.params['n_it']):
            w1 = W.reshape((n_dim * self.n_classes))
            G = self.gradLoss(w1, X, T).reshape((n_dim, self.n_classes))
            W -= self.params['rho'] * G

        return W

    def fit(self, X, Y):
        """
        Fits a logistic regression model to instances in X given the labels in
        Y

        Parameters
        ----------
        X: numpy.ndarray(n_samples, n_features)
            Input data
        Y: numpy.array(nsamples)
            Target for X. Each target can be a index in
            [0,..., self.n_classes-1] or a binary vector with dimension
            self.n_classes

        Returns:
            :self
        """

        self.n_dim = X.shape[1]

        # If labels are 1D, transform them into binary label vectors
        if len(Y.shape) == 1:

            # If the alphabet is not [0, 1, ..., n_classes-1] transform
            # labels into these values.
            # if not(set(self.classes_) < set(xrange(self.n_classes))):
            #     alphabet_inv = dict(zip(self.classes_,range(self.n_classes)))
            #     Y0 = np.array([alphabet_inv[c] for c in Y])
            # else:
            #     Y0 = Y

            T = self.index2bin(Y, self.n_classes)

        else:
            T = Y

        # Optimization
        if self.optimizer == 'GD':
            self.W = self.gd(X, T)
        else:
            w0 = 1 * np.random.randn(X.shape[1] * self.n_classes)
            res = sp.optimize.minimize(
                self.loss, w0, args=(X, T), method=self.optimizer,
                jac=self.gradLoss, hess=None, hessp=None, bounds=None,
                constraints=(), tol=None, callback=None,
                options={'disp': False, 'gtol': 1e-20,
                         'eps': 1.4901161193847656e-08, 'return_all': False,
                         'maxiter': None, 'norm': np.inf})
            #    options=None)
            self.W = res.x.reshape((self.n_dim, self.n_classes))

            # if res.status != 0:
            #     print "{0}-{1}: Status {2}. {3}. {4}".format(
            #         self.method, self.optimizer, res.status, res.success,
            #         res.message)
            # wtest = res.x
            # error = sp.optimize.check_grad(
            #     self.logLoss, self.gradLogLoss, wtest, X, T)
            # print "Check-grad error = {0}".format(error)

        return     # self    # w, nll_tr

    def predict(self, X):

        # Class
        D = np.argmax(X @ self.W, axis=1)

        return D  # p, D

    def predict_proba(self, X):

        # Linear term
        V = X @ self.W

        if self.canonical_link:
            if self.params['loss'] == 'CE':
                # Compute posterior class probabilities for weights w.
                p = self.softmax(V)

            elif self.params['loss'] == 'LBL':

                k2 = self.params['k'] / 2
                alpha = self.params['alpha']
                beta = self.params['beta']

                # Compute posterior class probabilities for weights w.
                # Forze each row in V to lie in the orthogonal subspace:
                # V = X W - (X·W·1) 1'   (where 1 is a c-ones vector)
                V -= np.mean(V, axis=1, keepdims=True)
                p = self.softmax(V)

                # Correction term
                delta = np.abs(V)**(beta - 1) * np.sign(V)
                delta -= np.sum(delta, axis=1, keepdims=True) / self.n_classes
                delta = k2 * beta * delta
                p = p + delta

            elif self.params['loss'] == 'square':
                exit("ERROR: No probabilistic predictions have been"
                     "implemented for the square error and canonical link")

        else:
            # Compute posterior class probabilities for weights w.
            p = self.softmax(V)
            # Old version. I have removed np.c_. Not sure if it is needed
            # p = (np.c_[self.softmax(X @ self.W)])

        return p

    def get_params(self, deep=True):

        # suppose this estimator has parameters "alpha" and "recursive"
        return {"n_classes": self.n_classes, "method": self.method,
                "optimizer": self.optimizer, "sound": self.sound,
                "params": self.params}
