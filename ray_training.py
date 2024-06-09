import ray

# Initialize Ray
ray.init()
temporary = {"test": 0}
# Define a remote function
@ray.remote
def square(x):
    return x * x *temporary["test"]

# Execute the function on the cluster
results = ray.get([square.remote(i) for i in range(4)])

# Print the results
print(results)

# Shutdown Ray
ray.shutdown()
