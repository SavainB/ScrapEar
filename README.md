
# **Prédiction de Notes au Piano à l'aide de TensorFlow**

## **Description**
Ce projet utilise l'apprentissage profond pour prédire les notes de piano à partir de fichiers audio. Le système convertit des fichiers audio en spectrogrammes, qui sont ensuite utilisés pour entraîner un modèle de réseau de neurones convolutionnel (CNN) basé sur TensorFlow. L'ensemble du processus suit un cycle complet de développement de modèle : préparation des données, création et entraînement du modèle, et évaluation des performances.

---

## **Fonctionnalités**
- Génération automatique de spectrogrammes à partir de fichiers audio (`.mp3` ou `.wav`).
- Réseau neuronal convolutionnel pour la classification des notes.
- Visualisation des métriques de performance (précision, perte, etc.). - A faire
- Sauvegarde et rechargement du modèle entraîné pour les prédictions futures. - A faire

---

## **Structure du Projet**
```plaintext
.
├── dataset/
│   ├── A/
│   │   ├── spectrogram1.png
│   │   ├── spectrogram2.png
│   ├── B/
│   ├── ...
├── song/
│   ├── example1.mp3
│   ├── example2.wav
├── mon_modele.h5
├── README.md
├── main.ipynb
```

- **`dataset/`** : Contient les images de spectrogrammes organisées par classe (notes A, B, C, etc.).
- **`song/`** : Dossier avec les fichiers audio bruts.
- **`mon_modele.h5`** : Modèle entraîné sauvegardé.
- **`main.ipynb`** : Notebook contenant le code principal du projet.
- **`README.md`** : Documentation du projet.

---

## **Prérequis**
### **Outils et bibliothèques nécessaires**
- Python 3.8+
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Librosa

### **Installation des dépendances**
```bash
pip install tensorflow numpy pandas matplotlib librosa
```

---

## **Étapes du Projet**

### 1. **Préparation des Données**
- Convertir les fichiers audio en spectrogrammes.
- Organiser les spectrogrammes en sous-dossiers par classe de notes (`A`, `B`, ..., `G`).

### 2. **Création du Modèle**
- Construire un réseau neuronal convolutionnel avec :
  - Plusieurs couches convolutionnelles (`Conv2D`).
  - Couches de pooling (`MaxPooling2D`).
  - Couches denses pour la classification.
  
### 3. **Entraînement**
- Utiliser un générateur de données (`ImageDataGenerator`) pour charger et normaliser les spectrogrammes.
- Configurer le modèle avec l'optimiseur Adam et la fonction de perte `categorical_crossentropy`.
- Évaluer la performance sur un ensemble de validation.

### 4. **Évaluation**
- Générer des courbes de perte et de précision.
- Tester le modèle sur un fichier audio de test.

---

## **Utilisation**

### **1. Génération de spectrogrammes**
Placez vos fichiers audio dans le dossier `song/`, puis exécutez le script pour générer les spectrogrammes :
```python
python generate_spectrograms.py
```

### **2. Entraînement**
Organisez vos spectrogrammes dans des sous-dossiers correspondant aux classes (`dataset/`), puis lancez l'entraînement :
```python
python train_model.py
```

### **3. Prédiction**
Testez un fichier audio pour prédire la note :
```python
python main.py 
```

## **Exemples**
### Exemple de Prédiction :
Fichier audio : `cnote.mp3`

```plaintext
La note prédite est : C
```

