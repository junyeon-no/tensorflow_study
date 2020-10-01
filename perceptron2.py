import tensorflow as tf

x = tf.random.uniform((10,5))
w = tf.random.uniform((5,3))
d = tf.matmul(x,w)

print(f'x와 w의 벡터 내적의 겨로가 크기 : {d.shape}')