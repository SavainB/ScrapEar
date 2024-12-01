from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import numpy  as np
# Charger le modèle pré-entrainé
model = load_model('mon_modele.h5')  # Assure-toi d'avoir sauvegardé ton modèle après l'entraînement
# Charger l'image du spectrogramme
audio_path = 'cnote.mp3'  # Remplace par le chemin de ton fichier audio
y, sr = librosa.load(audio_path, sr=None)  # `y` est le tableau des données audio, `sr` est la fréquence d’échantillonnage

# Calcul du spectrogramme
S = np.abs(librosa.stft(y))  # Transformée de Fourier pour obtenir les fréquences

# Affichage du spectrogramme
plt.figure(figsize=(10, 6))
librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), sr=sr, y_axis='log', x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogramme')
plt.xlabel('Temps')
plt.ylabel('Fréquence')

# Sauvegarde du spectrogramme en PNG
plt.savefig("spectrogramme.png")
img_path = 'spectrogramme.png'  # Remplace par ton chemin de fichier
img = image.load_img(img_path, target_size=(256, 256))  # Adapter la taille selon ton modèle
img_array = image.img_to_array(img)  # Convertir l'image en un tableau numpy
img_array = np.expand_dims(img_array, axis=0)  # Ajouter une dimension pour batch_size (shape: (1, 256, 256, 3))

# Normaliser l'image (si c'est ce que tu as fait pendant l'entraînement)
img_array = img_array / 255.0  # Par exemple, si tu as divisé par 255 lors de l'entraînement

# Faire la prédiction
predictions = model.predict(img_array)

# Trouver la classe avec la plus haute probabilité
predicted_class = np.argmax(predictions, axis=1)
print(predicted_class)
# Obtenir le nom de la classe prédite (par exemple, une note musicale)
class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G']  # Remplace par les classes que tu as utilisées
predicted_note = class_names[predicted_class[0]]

print(f"La note prédite est : {predicted_note}")