# Projet Final IA Générative Sorbonne

## Objectif du Projet

Ce projet vise à construire un modèle de Langage Large (LLM) basé sur la méthode RAG (Génération Augmentée de Récupération) pour améliorer les résultats de l'IA générative dans la création de quiz. L'objectif est de permettre à l'utilisateur de spécifier des informations précises pour générer un contexte de quiz adapté.

## Fonctionnalités

L'utilisateur doit fournir les informations suivantes pour générer le quiz :

1. **Nom du sujet** : Choisir parmi les options suivantes :
   - Aide humanitaire
   - Crise humanitaire
   - Droit fondamental
   - Droit Civil
   - Droit International

2. **Option des réponses** : Définir si les questions auront des choix uniques ou multiples.

3. **Nombre de tokens** : Spécifier le nombre de tokens pour générer le quiz.

4. **Nombre de questions** : Indiquer le nombre de questions à générer (si possible).

## Structure du Quiz en Sortie

Le quiz généré aura la structure suivante :

- **Nom du sujet** : Trouvé dans les documents.
- **Nouvelle question** : Générée en fonction du contexte.
- **Nouvelle liste des options** : Liste des choix possibles pour la question.
- **Une ou plusieurs nouvelles réponses correctes** : Indiquer la ou les réponses exactes.
- **Nouvelle explication** : Fournir une explication pour comprendre la ou les réponses exactes.

## Utilisation

Pour utiliser ce script, suivez les étapes suivantes :

1. **Installer les dépendances nécessaires** :
   ```bash
   pip install -r requirements.txt
   ```

2. **Exécuter le script** :
   ```bash
   python app.py
   ```

3. **Fournir les informations requises** :
   - Entrer le nom du sujet.
   - Choisir l'option des réponses (uniques ou multiples).
   - Spécifier le nombre de tokens.
   - Indiquer le nombre de questions à générer.


## Exemple de Sortie

```plaintext
Nom du sujet : Aide humanitaire

Question 1 : Quelle est la principale organisation internationale chargée de coordonner l'aide humanitaire ?
Options :
A. Organisation des Nations Unies (ONU)
B. Croix-Rouge
C. Médecins Sans Frontières (MSF)
D. Organisation mondiale de la Santé (OMS)
Réponse correcte : A
Explication : L'ONU est la principale organisation internationale chargée de coordonner l'aide humanitaire à travers diverses agences comme le PAM et l'UNHCR.

Question 2 : Quel est le principal défi rencontré dans l'aide humanitaire ?
Options :
A. Manque de financement
B. Accès aux zones de crise
C. Coordination des efforts
D. Tous les above
Réponse correcte : D
Explication : Tous ces défis sont fréquemment rencontrés dans l'aide humanitaire, rendant la coordination et la fourniture d'aide complexes.
```

## Contribution

Les contributions sont les bienvenues ! Veuillez ouvrir une issue ou soumettre une pull request pour toute amélioration ou correction.

## Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

---

Ce README fournit une vue d'ensemble du projet et des instructions pour son utilisation. Pour plus de détails, référez-vous aux commentaires dans le code source.