import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Étape 2 : Chargement des données
# Utilisation de ImageDataGenerator pour charger les spectrogrammes
train_datagen = ImageDataGenerator(rescale=1./255)  # Normalisation des images
train_generator = train_datagen.flow_from_directory(
    'dataset/',  # Répertoire contenant les sous-dossiers de notes
    target_size=(256, 256),  # Taille des images
    batch_size=32,
    class_mode='categorical'  # Utilisation de la classification catégorielle
)

# Étape 3 : Création du modèle
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(train_generator.num_classes, activation='softmax')  # Nombre de classes basé sur les sous-dossiers
])

# Compilation du modèle
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Étape 4 : Entraînement du modèle
model.fit(train_generator, epochs=10)
# Sauvegarder le modèle entraîné
model.save('mon_modele.h5')  # Le fichier .h5 contient l'architecture, les poids et la configuration du modèle
