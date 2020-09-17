import tensorflow as tf
import numpy as np

a = tf.constant(3)
b = tf.constant(2)

print(tf.add(a,b))
print(tf.subtract(a,b))

print(tf.multiply(a,b).numpy())
print(tf.divide(a,b).numpy())

c = tf.add(a,b)

c_square = np.square(c, dtype= np.float32)

c_tensor = tf.convert_to_tensor(c_square)

print('numpy array : %0.1f, applying square with numpy : %0.1f, convert_to_tensro : %0.1f' % (c, c_square, c_tensor))