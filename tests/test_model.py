import numpy as np
import unittest
from tensorflow.keras.models import load_model

class TestModel(unittest.TestCase):
       def setUp(self):
           self.model = load_model("models/chest_xray_model.h5")
           self.input_shape = (224, 224, 1)

       def test_model_output_shape(self):
           test_input = np.random.rand(1, 224, 224, 1).astype(np.float32)
           output = self.model.predict(test_input)
           self.assertEqual(output.shape, (1, 14))

       def test_model_compile(self):
           self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
           self.assertTrue(self.model.optimizer)
           self.assertTrue(self.model.loss)
           self.assertTrue(len(self.model.metrics) > 0)

if __name__ == '__main__':
       unittest.main()