import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import column_or_1d
from sklearn.utils._encode import _unique, _encode
from sklearn.utils.validation import _num_samples, check_array, check_is_fitted

class LabelEncoderManual(TransformerMixin, BaseEstimator,):

    def __init__(self, mapping) -> None:
        """
        Arguments:
        ----------
        mapping - a dictionary like structure that defines mapping between old and new classes.
                    Old classes are used as keys to the dictionary.
        """
        super().__init__()
        self.mapping_ = mapping

    def fit(self, y):
        """Fit label encoder.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : returns an instance of self.
            Fitted label encoder.
        """
        y = column_or_1d(y, warn=True)
        tmp_classes = _unique(y)

        n_classes = len(tmp_classes)
        encoded_classes = np.asanyarray( [k for k in self.mapping_])
        n_encoded_classes = len(encoded_classes)
        if n_classes > n_encoded_classes:
            raise ValueError("The number of classes inside y ({}) is greater than the number of labels inside mapping ({}) ".format(n_classes, n_encoded_classes ))
        
        
        inv_map = { v:k for k, v in self.mapping_.items()}
        n_unique_target_labels = len(inv_map)
        if n_unique_target_labels  != n_encoded_classes:
            raise ValueError("The mapping is not 'one-to-one' mapping")
        
        self.classes_ = tmp_classes
        self.inverse_mapping_ = inv_map
        self.encoded_classes = np.sort( np.asanyarray([k for k in self.inverse_mapping_]))
        

        return self

    def fit_transform(self, y):
        """Fit label encoder and return encoded labels.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        y : array-like of shape (n_samples,)
            Encoded labels.
        """
        
        return self.fit(y).transform(y)

    def transform(self, y):
        """Transform labels to normalized encoding.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        y : array-like of shape (n_samples,)
            Labels as normalized encodings.
        """
        check_is_fitted(self)
        y = column_or_1d(y, dtype=self.classes_.dtype, warn=True)
        # transform of empty array is empty array
        if _num_samples(y) == 0:
            return np.array([])

        encoded = np.asanyarray([self.mapping_[y_i] for y_i in y ])

        return encoded

    def inverse_transform(self, y):
        """Transform labels back to original encoding.

        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            Original encoding.
        """
        check_is_fitted(self)
        y = column_or_1d(y, warn=True)
        # inverse transform of empty array is empty array
        if _num_samples(y) == 0:
            return np.array([])

        diff = np.setdiff1d(y, self.encoded_classes )
        if len(diff):
            raise ValueError("y contains previously unseen labels: %s" % str(diff))
        y = np.asarray(y)

        inv_encoded = np.asanyarray([self.inverse_mapping_[y_i] for y_i in y ])

        return inv_encoded

    def _more_tags(self):
        return {"X_types": ["1dlabels"]}


    