import unittest
from sklearn.utils.estimator_checks import check_transformer_general, check_estimator
from label_encoder_random.transformers.label_encoder_manual import LabelEncoderManual

from label_encoder_random.transformers.label_encoder_random import LabelEncoderRandom
from sklearn.datasets import load_iris
import numpy as np
from copy import deepcopy

class LabelEncoderRandomTest(unittest.TestCase):

    
    def generate_sets(self,n_labels=10, n_objects = 1000 ,  dtypes = [int, str, np.uint, np.dtype('U10'), np.dtype('S5')]):
         
         y_pre = np.random.choice(n_labels, size=n_objects)
         for dtype in dtypes:
             yield y_pre.astype(dtype)

    
    def test_simple(self):
        
        for y in self.generate_sets():

            rencoder = LabelEncoderRandom()
            rencoder.fit(y)

            mapping = rencoder.mapping_

            encoder = LabelEncoderManual(mapping)
           
            y_mod = encoder.fit_transform(y)

            self.assertIsNotNone(y_mod, "None Prediction")
            self.assertIsInstance(y_mod, np.ndarray, "Not an numpy array")
            self.assertTrue( len(np.unique(y)) == len(np.unique(y_mod)), "Wrong number of unique labels in encoded version")

            y_inv = encoder.inverse_transform(y_mod)

            self.assertTrue( np.all( y_inv == y ), "Wrong inverse transformation" )

    
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

    def test_wrong_mapping1(self):

        y = [1,2,3,1,2,3,1,2,3]
        
        try:
            wrong_map_1 = {1:"A", 2:"B" }
            enc = LabelEncoderManual(wrong_map_1)
            enc.fit(y)
            self.fail("ValueError should have been raised!")
        except ValueError as ve:
            print("Value error")
            pass
        except Exception as e:
            self.fail("Unexpected exception {}".format(e))

    def test_wrong_mapping2(self):

        y = [1,2,3,1,2,3,1,2,3]
        
        try:
            wrong_map_1 = {1:"A", 2:"B", 3:"A" }
            enc = LabelEncoderManual(wrong_map_1)
            enc.fit(y)
            self.fail("ValueError should have been raised!")
        except ValueError as ve:
            print("Value error")
            pass
        except Exception as e:
            self.fail("Unexpected exception {}".format(e))    


if __name__ == '__main__':
    unittest.main()