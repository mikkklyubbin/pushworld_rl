import wandb
import numpy as np
import os
import requests
print("HTTP_PROXY в окружении Python:", os.environ.get('HTTP_PROXY'))
print("HTTPS_PROXY в окружении Python:", os.environ.get('HTTPS_PROXY'))
response = requests.get('https://api.ipify.org?format=json', timeout=5)
print(f"\n прокси: {response.json()['ip']}")
wandb.init(project="my-project", name="experiment-1")


model_weights = np.random.randn(10, 10).tolist()  
model_bias = np.random.randn(10).tolist()       
config = {
    "learning_rate": 0.01,
    "layer1_weights_shape": np.array(model_weights).shape,
    "layer1_bias_shape": np.array(model_bias).shape
}
wandb.config.update(config)

for epoch in range(10):
    loss = 0.1 * (0.9 ** epoch)
    accuracy = 1.0 - loss
    wandb.log({"epoch": epoch, "loss": loss, "accuracy": accuracy})

wandb.finish()