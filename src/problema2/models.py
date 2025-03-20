import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        # Definir una red neuronal simple con una capa oculta
        self.hidden_size = 128  
        self.fc1 = nn.Linear(1, self.hidden_size)  # Capa oculta
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_size, 1)  # Capa de salida 

    def forward(self, x):
        # Paso hacia adelante en la red
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def get_loss(self, x, y_true):
        # Calcular la pérdida de error cuadrático medio
        y_pred = self.forward(x)
        loss_fn = nn.MSELoss()
        return loss_fn(y_pred, y_true)

    def train_model(self, train_data, num_epochs=1000, learning_rate=0.01):
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            loss = self.get_loss(train_data["x"], train_data["y"])
            loss.backward()
            optimizer.step()
            
            if epoch % 100 == 0:
                print(f"Época {epoch}, Pérdida: {loss.item()}")

        print("Entrenamiento completado.")

# Generar datos de entrenamiento
x_train = np.linspace(-2 * np.pi, 2 * np.pi, 100).reshape(-1, 1)
y_train = np.sin(x_train)

# Convertir a tensores de PyTorch
x_tensor = torch.tensor(x_train, dtype=torch.float32)
y_tensor = torch.tensor(y_train, dtype=torch.float32)

# Crear el modelo y entrenarlo
model = RegressionModel()
train_data = {"x": x_tensor, "y": y_tensor}
model.train_model(train_data)

x_test = np.linspace(-2 * np.pi, 2 * np.pi, 100).reshape(-1, 1)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)

# Hacer predicciones
y_pred = model.forward(x_test_tensor).detach().numpy()

# Graficar resultados
plt.scatter(x_train, y_train, label="Datos reales", color="blue")
plt.plot(x_test, y_pred, label="Predicciones", color="red")
plt.legend()
plt.show()