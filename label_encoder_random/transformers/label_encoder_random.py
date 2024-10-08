import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import column_or_1d
from sklearn.utils._encode import _unique, _encode
from sklearn.utils.validation import _num_samples, check_array, check_is_fitted

class LabelEncoderRandom(TransformerMixin, BaseEstimator,):

    def __init__(self,offset=0, randomize=True) -> None:
        super().__init__()
        self.offset = offset
        self.randomize = randomize

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
        self.classes_ = _unique(y)
        
        self.mapping_, self.inverse_mapping_ = LabelEncoderRandom.generate_random_mapping(y, self.offset, self.randomize)
        self.encoded_classes = np.sort( np.asanyarray([k for k in self.inverse_mapping_]))

        return self
    
    @staticmethod
    def generate_random_mapping(y, offset=0, randomize=True):
        """
        Generates random mapping and inverse mapping

        Arguments:
        ----------
        y: containing labels
        Returns:
        (mapping:dict, inverse_mapping:dict)
        """
        classes_ = _unique(y)
        n_classes = len(classes_)
        encoded_classes = np.arange(n_classes) + offset
        if randomize:
            np.random.shuffle(encoded_classes)
        mapping_ = { orig_class:encoded_class for orig_class, encoded_class in zip ( classes_,  encoded_classes )}
        inverse_mapping_ = { v:k for k, v in mapping_.items()}

        return mapping_, inverse_mapping_


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

        diff = np.setdiff1d(y, self.encoded_classes)
        if len(diff):
            raise ValueError("y contains previously unseen labels: %s" % str(diff))
        y = np.asarray(y)

        inv_encoded = np.asanyarray([self.inverse_mapping_[y_i] for y_i in y ])

        return inv_encoded

    def _more_tags(self):
        return {"X_types": ["1dlabels"]}


    