import tensorflow as tf
from tensorflow.keras import models, layers

# 1. Initialisation du modèle séquentiel
model = models.Sequential()

# 2. Première couche convolutive + MaxPooling
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(layers.MaxPooling2D((2, 2)))

# 3. Deuxième couche convolutive pour approfondir l'extraction
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# 4. Troisième couche convolutive pour les détails complexes
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# 5. Aplatissement de la matrice (Transformation en vecteur 1D)
model.add(layers.Flatten())

# 6. Couche Dense intermédiaire (Fully Connected)
model.add(layers.Dense(128, activation='relu'))

# 7. Couche de sortie avec 7 neurones (pour les 7 émotions de FER-2013)
# L'activation Softmax permet d'obtenir des probabilités entre 0 et 1
model.add(layers.Dense(7, activation='softmax'))

# Affichage du résumé de l'architecture dans le terminal
model.summary()

# 8. Compilation du modèle avec les hyperparamètres
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 9. Exemple de commande pour l'entraînement (si exécuté sur le dataset)
# history = model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val))

# 10. Sauvegarde du modèle entraîné au format standard H5
model.save('emotion_model.h5')
print("Modèle sauvegardé avec succès sous le nom 'emotion_model.h5'")