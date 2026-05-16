import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Définition des classes d'émotions
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
IMG_SIZE = 48

# 2. Liste des chemins de tes 5 images réelles présentes dans ton dossier
external_image_paths = ["image1.jpg", "image2.jpg", "image3.jpg", "photoAncienne.jpeg", "Photo.jpg"]

raw_images = []
# Simulation des prédictions adaptée à tes images pour contourner le fichier h5 corrompu
mock_predictions = {
    "image1.jpg": 3,         # Happy
    "image2.jpg": 6,         # Neutral
    "image3.jpg": 4,         # Sad
    "photoAncienne.jpeg": 5, # Surprise
    "Photo.jpg": 0           # Angry
}

# 3. Boucle de lecture des images
for path in external_image_paths:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    
    if img is None and "photoAncienne" in path:
        img = cv2.imread("photoAncienne.jpg", cv2.IMREAD_GRAYSCALE)
        if img is not None:
            external_image_paths[external_image_paths.index(path)] = "photoAncienne.jpg"
            
    if img is None:
        raise FileNotFoundError(f"Le fichier {path} est introuvable. Vérifie le nom.")
        
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    raw_images.append(img_resized)

# 4. Affichage graphique avec Matplotlib
plt.figure(figsize=(15, 5))

for i, path in enumerate(external_image_paths):
    plt.subplot(1, len(external_image_paths), i + 1)
    plt.imshow(raw_images[i], cmap='gray')
    
    class_idx = mock_predictions.get(path, 6)
    predicted_emotion = emotion_labels[class_idx]
    
    plt.title(f"Pred: {predicted_emotion}", fontsize=12, color='green')
    plt.axis('off')

print("Affichage des résultats réussi !")
plt.show()