# TP : Reconnaissance des Émotions Faciales (Deep Learning & Computer Vision)

Ce dépôt contient l'implémentation pratique du **TP6** dédié à l'apprentissage automatique et à la classification des émotions à partir d'images faciales réelles, basé sur le framework **TensorFlow** et la bibliothèque **OpenCV**.

---

## 🚀 Fonctionnalités du Projet
* **Prétraitement des données :** Conversion dynamique des images réelles en niveaux de gris et redimensionnement automatique au format standard $48 \times 48$ pixels (conforme au dataset FER-2013).
* **Pipeline d'Inférence :** Chargement des scripts et exécution des prédictions logiques sur des visages réels.
* **Visualisation Graphique :** Affichage soigné de 5 visages de test alignés avec leurs prédictions respectives affichées en vert (`Happy`, `Neutral`, `Sad`, `Surprise`, `Angry`) via **Matplotlib**.

---

## 📁 Structure du Projet
L'arborescence du dépôt se présente comme suit :
* `test_images.py` : Script principal pour le traitement des images, l'inférence logicielle et l'affichage des résultats.
* `train_emotion.py` : Script dédié à la configuration et à la logique d'entraînement du modèle.
* `Photo.jpg`, `image1.jpg`, `image2.jpg`, `image3.jpg`, `photoAncienne.jpeg` : Les 5 images réelles utilisées pour valider le modèle de test.

---

## 🛠️ Difficultés Rencontrées & Solutions Adoptées

Lors du développement en environnement local (VS Code), plusieurs défis techniques ont été surmontés :

1. **Version de Python & TensorFlow :** La version par défaut du système étant trop récente, un environnement virtuel stable sous **Python 3.11.7** a été configuré pour garantir la compatibilité complète des frameworks de Deep Learning.
2. **Gestion de l'Environnement Local :** Face aux restrictions de sécurité PowerShell (`PSSecurityException`) et aux faux positifs de l'anti-virus lors de l'exécution, les autorisations système ont été ajustées manuellement pour exécuter le code en toute sécurité.
3. **Alternative Cloud (Google Colab) :** En tant que solution d'ingénierie logicielle pour contourner les lourdes contraintes matérielles locales (RAM/GPU), l'utilisation de **Google Colab** a été adoptée comme alternative cloud idéale pour l'exécution fluide de TensorFlow sans configuration locale.

---

## 💻 Technologies Utilisées
* **Language :** Python 3.11
* **Deep Learning Framework :** TensorFlow / Keras
* **Computer Vision :** OpenCV (cv2)
* **Data Visualization :** Matplotlib
* **Version Control :** Git / GitHub
