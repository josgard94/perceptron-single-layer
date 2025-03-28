## 📄 `README.es.md` (versión en español)

> 🇬🇧 [Read in English](README.md)

# 🧠 Perceptrón de Capa Única

¡Bienvenido a este clásico proyecto de aprendizaje automático! Este repositorio contiene una implementación en **Python** del **Perceptrón de una sola capa**, también conocido como **Red Neuronal Simple**, propuesto por **Frank Rosenblatt** en 1958 e inspirado en el modelo de neuronas artificiales de **McCulloch y Pitts** (1943).

---

## 📘 Descripción

El perceptrón simula el funcionamiento básico de una neurona del cerebro:

- Recibe entradas 🧾 (`x`)
- Cada entrada se multiplica por un **peso** 📊 (`w`)
- Se aplica una **función de activación** (`z = w·x + b`)
- Si el resultado supera un cierto **umbral** (`θ`), la neurona se **activa** 🔥

Este algoritmo es un **clasificador lineal binario**, usado para **aprendizaje supervisado**.

---

## ⚙️ Características

- 🐍 Implementado en **Python**
- 🎯 Realiza **clasificación binaria**
- 📈 Entrenamiento paso a paso con actualización de pesos
- 🧪 Visualización de resultados

---

## 📸 Resultados

### 🔹 Frontera de Decisión
![Frontera de Decisión](https://github.com/josgard94/perceptron-single-layer/blob/main/screenshots/result1.png)

### 🔹 Proceso de Aprendizaje
![Proceso de Aprendizaje](https://github.com/josgard94/perceptron-single-layer/blob/main/screenshots/result2.png)

---

## 🚀 Cómo Ejecutar

Asegúrate de tener Python 3 y las dependencias necesarias:

```bash
pip install -r requirements.txt
```
Luego ejecutas el script principal:
```bash
python main.py
```
## 📚 Referencias
- Rosenblatt, F. (1958). The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain.
- McCulloch, W. S., & Pitts, W. (1943). A Logical Calculus of the Ideas Immanent in Nervous Activity.

## ⭐ ¡Dale una estrella!
Si este proyecto te resultó útil o interesante, ¡no olvides dejar una ⭐ en el repositorio!
