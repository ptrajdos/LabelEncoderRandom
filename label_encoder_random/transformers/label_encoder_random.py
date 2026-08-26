import numpy as np
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import column_or_1d
from sklearn.utils._encode import _unique
from sklearn.utils.validation import _num_samples, check_is_fitted


from packaging.version import parse as parse_version

_SK_VERSION = parse_version(sklearn.__version__)


class LabelEncoderRandom(
    TransformerMixin,
    BaseEstimator,
):
    """Encode target labels with random integer values.

    This transformer assigns a unique random integer to each unique label,
    producing a non-sequential encoding. It is useful when the ordinal
    relationship implied by sequential label encoding is undesirable.

    Parameters
    ----------
    offset : int, default=0
        Value added to every encoded label so that the encoded range
        starts from ``offset`` instead of 0.
    randomize : bool, default=True
        If ``True``, the integer codes assigned to labels are randomly
        shuffled. If ``False``, labels are encoded sequentially (like
        the standard ``LabelEncoder``).
    disable_check : bool, default=False
        If ``True``, ``inverse_transform`` skips the validation that
        checks whether all values in *y* were seen during fitting.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Unique classes found during ``fit``.
    mapping_ : dict
        Dictionary mapping original labels to encoded integers.
    inverse_mapping_ : dict
        Dictionary mapping encoded integers back to original labels.
    encoded_classes : ndarray of shape (n_classes,)
        Sorted array of encoded integer values.

    Examples
    --------
    >>> from label_encoder_random.transformers.label_encoder_random import LabelEncoderRandom
    >>> le = LabelEncoderRandom(randomize=False)
    >>> le.fit(["cat", "dog", "cat", "bird"])
    LabelEncoderRandom(randomize=False)
    >>> le.classes_
    array(['bird', 'cat', 'dog'], dtype='<U4')
    >>> le.transform(["cat", "dog"])
    array([1, 2])
    >>> le.inverse_transform([1, 2])
    array(['cat', 'dog'], dtype=object)
    """

    def __init__(self, offset=0, randomize=True, disable_check=False) -> None:
        super().__init__()
        self.offset = offset
        self.randomize = randomize
        self.disable_check = disable_check

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

        self.mapping_, self.inverse_mapping_ = (
            LabelEncoderRandom.generate_random_mapping(y, self.offset, self.randomize)
        )
        self.encoded_classes = np.sort(
            np.asanyarray([k for k in self.inverse_mapping_])
        )
        
        # Create sorted arrays for efficient lookup in transform
        sorted_indices = np.argsort(self.classes_)
        self.sorted_classes_ = self.classes_[sorted_indices]
        self.sorted_encoded_classes_ = np.asarray([self.mapping_[c] for c in self.sorted_classes_])
        
        # Create direct array lookup for inverse_transform
        # encoded_classes are contiguous integers from offset to offset+n_classes-1
        min_encoded = np.min(self.encoded_classes)
        max_encoded = np.max(self.encoded_classes)
        self._inverse_array = np.empty(max_encoded - min_encoded + 1, dtype=object)
        for encoded, original in self.inverse_mapping_.items():
            self._inverse_array[encoded - min_encoded] = original
        self._min_encoded = min_encoded

        return self

    @staticmethod
    def generate_random_mapping(y, offset=0, randomize=True):
        """Generate a random mapping and its inverse for the given labels.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target labels from which unique classes are extracted.
        offset : int, default=0
            Value added to every encoded label so that the encoded range
            starts from ``offset`` instead of 0.
        randomize : bool, default=True
            If ``True``, the integer codes are randomly shuffled;
            otherwise they are assigned sequentially.

        Returns
        -------
        mapping : dict
            Dictionary mapping original labels to encoded integers.
        inverse_mapping : dict
            Dictionary mapping encoded integers back to original labels.
        """
        classes_ = _unique(y)
        n_classes = len(classes_)
        encoded_classes = np.arange(n_classes) + offset
        if randomize:
            np.random.shuffle(encoded_classes)
        mapping_ = {
            orig_class: encoded_class
            for orig_class, encoded_class in zip(classes_, encoded_classes)
        }
        inverse_mapping_ = {v: k for k, v in mapping_.items()}

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

        # Use searchsorted for efficient O(log n) lookup instead of dict
        indices = np.searchsorted(self.sorted_classes_, y)
        encoded = self.sorted_encoded_classes_[indices]

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

        if not self.disable_check:
            diff = np.setdiff1d(y, self.encoded_classes)
            if len(diff):
                raise ValueError("y contains previously unseen labels: %s" % str(diff))
        y = np.asarray(y)

        # Direct array indexing for O(1) lookup
        indices = y - self._min_encoded
        inv_encoded = self._inverse_array[indices]

        return inv_encoded

    def _more_tags(self):
        return {"X_types": ["1dlabels"]}

    if _SK_VERSION >= parse_version("1.6"):

        def __sklearn_tags__(self):
            tags = super().__sklearn_tags__()
            tags.non_deterministic = False
            return tags
