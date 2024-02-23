#Install tf2onnx, onnx, tensorflow
import onnx
import os
import tf2onnx

from tensorflow.keras.models import load_model

model = load_model('models/model-040.h5')
onnx_model, _ = tf2onnx.convert.from_keras(model)
onnx.save(onnx_model, 'model.onnx')