import unittest
from sklearn.utils.estimator_checks import check_transformer_general, check_estimator

from label_encoder_random.transformers.label_encoder_random import LabelEncoderRandom
from sklearn.datasets import load_iris
import numpy as np
from copy import deepcopy

class LabelEncoderRandomTest(unittest.TestCase):

    def get_encoders(self):
        return [
            LabelEncoderRandom(),
            LabelEncoderRandom(offset=1),
            LabelEncoderRandom(offset=1 << 16),
        ]
    
    def generate_sets(self,n_labels=10, n_objects = 1000 ,  dtypes = [int, str, np.uint, np.dtype('U10'), np.dtype('S5')]):
         
         y_pre = np.random.choice(n_labels, size=n_objects)
         for dtype in dtypes:
             yield y_pre.astype(dtype)

    
    def test_simple(self):
        
        for y in self.generate_sets():

            for encoder in self.get_encoders():
                encoder_copy = deepcopy(encoder)
                encoder.fit(y)
                y_mod = encoder.transform(y)

                self.assertIsNotNone(y_mod, "None Prediction")
                self.assertIsInstance(y_mod, np.ndarray, "Not an numpy array")
                self.assertTrue( len(np.unique(y)) == len(np.unique(y_mod)), "Wrong number of unique labels in encoded version")

                y_inv = encoder.inverse_transform(y_mod)

                self.assertTrue( np.all( y_inv == y ), "Wrong inverse transformation" )

        
                y_mod_c = encoder_copy.fit_transform(y)
                self.assertFalse( np.allclose(y_mod, y_mod_c), "Transformed label assignment is not random!")

                y_empty = encoder.transform([])
                self.assertIsInstance(y_empty, np.ndarray, "Transforming empty data, wrong output type")
                self.assertTrue( len(y_empty) == 0, "Transforming empty data, wrong output length")

                y_inv_empty = encoder.inverse_transform([])
                self.assertIsInstance(y_inv_empty, np.ndarray, "Transforming empty data, wrong output type")
                self.assertTrue( len(y_inv_empty) == 0, "Transforming empty data, wrong output length")

                try:
                    y_wr = encoder.inverse_transform([12345])
                    self.fail("ValueError should have been raised!")
                except ValueError:
                    pass
                except Exception as e:
                    self.fail("Wrong exception: {}".format(e))

if __name__ == '__main__':
    unittest.main()