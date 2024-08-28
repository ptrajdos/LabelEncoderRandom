import unittest
from sklearn.utils.estimator_checks import check_transformer_general, check_estimator
from label_encoder_random.transformers.label_encoder_manual import LabelEncoderManual

from label_encoder_random.transformers.label_encoder_random import LabelEncoderRandom
from sklearn.datasets import load_iris
import numpy as np
from copy import deepcopy

class LabelEncoderManualTest(unittest.TestCase):

    
    def generate_sets(self,n_labels=10, n_objects = 1000 ,  dtypes = [int, str, np.uint, np.dtype('U10'), np.dtype('S5')]):
         
         y_pre = np.random.choice(n_labels, size=n_objects)
         for dtype in dtypes:
             yield y_pre.astype(dtype)


    def generate_sets_manu(self, base_labels=['A','B','C'], n_objects = 1000):
        pass

    def gen_rencoders(self):
        return[
            LabelEncoderRandom(),
            LabelEncoderRandom(offset=1),
            LabelEncoderRandom(offset=1<<16),
        ]

    def test_manual_numeric(self):
        for y in self.generate_sets([1,3,5], dtypes=[int, np.uint]):

            mapping = {
                1:-1,
                3:-3,
                5:-5
            }

            encoder = LabelEncoderManual(mapping)
        
            y_mod = encoder.fit_transform(y)

            self.assertIsNotNone(y_mod, "None Prediction")
            self.assertIsInstance(y_mod, np.ndarray, "Not an numpy array")
            self.assertTrue( len(np.unique(y)) == len(np.unique(y_mod)), "Wrong number of unique labels in encoded version")

            y_inv = encoder.inverse_transform(y_mod)

            self.assertTrue( np.all( y_inv == y ), "Wrong inverse transformation" )

    def test_manual_str(self):
        for y in self.generate_sets(['A', 'B', 'C'], dtypes=[np.str_]):

            mapping = {
                'A':'D',
                'B':'E',
                'C':'f',
            }

            encoder = LabelEncoderManual(mapping)
        
            y_mod = encoder.fit_transform(y)

            self.assertIsNotNone(y_mod, "None Prediction")
            self.assertIsInstance(y_mod, np.ndarray, "Not an numpy array")
            self.assertTrue( len(np.unique(y)) == len(np.unique(y_mod)), "Wrong number of unique labels in encoded version")

            y_inv = encoder.inverse_transform(y_mod)

            self.assertTrue( np.all( y_inv == y ), "Wrong inverse transformation" )

    
    def test_simple(self):
        
        for rencoder in self.gen_rencoders():
            for y in self.generate_sets():

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

    def test_mapping_larger_map(self):

        y = [1,2,3,1,2,3,1,2,3]
        map_1 = {1:"A", 2:"B", 3:"C", 4:"D", 5:"E" }
        encoder = LabelEncoderManual(map_1)

        y_mod = encoder.fit_transform(y)

        self.assertIsNotNone(y_mod, "None Prediction")
        self.assertIsInstance(y_mod, np.ndarray, "Not an numpy array")
        self.assertTrue( len(np.unique(y)) == len(np.unique(y_mod)), "Wrong number of unique labels in encoded version")

        y_inv = encoder.inverse_transform(y_mod)

        self.assertTrue( np.all( y_inv == y ), "Wrong inverse transformation" )


        y_i = ['D', "E", "E", "D"]
        #This should work in that way
        y_m = encoder.inverse_transform(y_i)

        self.assertIsNotNone(y_m, "None Prediction")
        self.assertIsInstance(y_m, np.ndarray, "Not an numpy array")
        self.assertTrue( len(np.unique(y_i)) == len(np.unique(y_m)), "Wrong number of unique labels in encoded version")

        
        


if __name__ == '__main__':
    unittest.main()