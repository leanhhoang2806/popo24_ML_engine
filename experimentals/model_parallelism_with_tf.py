import tensorflow as tf
import os
import json
import logging
from tqdm import tqdm

# Set logging level to info
tf.get_logger().setLevel('INFO')

# Configure Python logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Read TF_CONFIG from environment variable
tf_config = os.environ.get('TF_CONFIG')
if tf_config:
    tf_config = json.loads(tf_config)
    cluster_spec = tf_config['cluster']
    task_type = tf_config['task']['type']
    task_index = tf_config['task']['index']
else:
    raise ValueError("TF_CONFIG environment variable is not set")

logger.info("TF_CONFIG: %s", tf_config)

# Create a cluster resolver from the cluster specification
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

logger.info("Starting training...")

# Custom training loop with tqdm for progress bar
epochs = 5
steps_per_epoch = len(train_images) // 32
validation_steps = len(test_images) // 32

for epoch in range(epochs):
    logger.info(f"Epoch {epoch + 1}/{epochs}")
    with tqdm(total=steps_per_epoch, desc=f"Epoch {epoch + 1}/{epochs}") as pbar:
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            model.train_on_batch(x_batch_train, y_batch_train)
            pbar.update(1)
            if step >= steps_per_epoch:
                break

    val_loss, val_acc = model.evaluate(test_dataset, steps=validation_steps)
    logger.info(f"Validation loss: {val_loss}, Validation accuracy: {val_acc}")

logger.info("Training finished")

# Check for GPU availability
if tf.config.list_physical_devices('GPU'):
    logger.info("TensorFlow is using the GPU.")
else:
    logger.info("TensorFlow is not using the GPU.")
