

"""
For 192.168.1.101 (Worker 0):
export TF_CONFIG='{
  "cluster": {
    "worker": ["192.168.1.101:2222", "192.168.1.102:2222"]
  },
  "task": {"type": "worker", "index": 0}
}'


For 192.168.1.102 (Worker 1):
export TF_CONFIG='{
  "cluster": {
    "worker": ["192.168.1.101:2222", "192.168.1.102:2222"]
  },
  "task": {"type": "worker", "index": 1}
}'

The script need to run on both machines
"""

import tensorflow as tf
import os
import json

# Set logging level to info
tf.get_logger().setLevel('INFO')


# Check for GPU availability
if tf.test.is_gpu_available():
    print("TensorFlow is using the GPU.")
else:
    print("TensorFlow is not using the GPU.")

# Read TF_CONFIG from environment variable
tf_config = os.environ.get('TF_CONFIG')
if tf_config:
    tf_config = json.loads(tf_config)
    cluster_spec = tf_config['cluster']
    task_type = tf_config['task']['type']
    task_index = tf_config['task']['index']
else:
    raise ValueError("TF_CONFIG environment variable is not set")

cluster_resolver = tf.distribute.cluster_resolver.SimpleClusterResolver(
    cluster_spec=tf.train.ClusterSpec(cluster_spec),
    task_type=task_type,
    task_id=task_index
)

strategy = tf.distribute.experimental.ParameterServerStrategy(cluster_resolver)

with strategy.scope():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0

train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(32)
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(32)

print("Starting training...")
model.fit(train_dataset, epochs=5, validation_data=test_dataset)


