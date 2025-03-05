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
Nom du sujet: Aide humanitaire

### Question 1: Qu'est-ce que l'aide humanitaire?
- a) La promotion du commerce entre différents pays
- b) L'assistance fournie lors de crises pour sauver des vies
- c) L'ensemble des cours dispensés dans les écoles
- d) Les services de santé réguliers fournis par des gouvernements

Réponse correcte: b
Explication: L'aide humanitaire est l'assistance apportée pour aider les personnes en cas de crises majeures telles que des catastrophes naturelles ou des conflits armés, avec le but principal de sauver des vies.

### Question 2: À quel moment l’aide humanitaire est-elle généralement fournie?
- a) Après des catastrophes naturelles
- b) Pendant des événements festifs
- c) Lors de la mise en œuvre de nouveaux programmes éducatifs
- d) Durant les périodes de stabilité économique

Réponse correcte: a
Explication: L’aide humanitaire est principalement fournie en réponse à des catastrophes naturelles, telles que des ouragans, des tremblements de terre ou des inondations, où les besoins des populations affectées sont immédiats.

### Question 3: Quel est l'objectif principal de l'aide humanitaire?
- a) Améliorer les relations diplomatiques
- b) Sauver des vies et soulager les souffrances
- c) Promouvoir le tourisme dans les zones affectées
- d) Encourager l'exportation de biens locaux

Réponse correcte: b
Explication: L'objectif principal de l'aide humanitaire est de sauver des vies et de soulager les souffrances des personnes touchées par des crises ou des catastrophes.
```

## Contribution

Les contributions sont les bienvenues ! Veuillez ouvrir une issue ou soumettre une pull request pour toute amélioration ou correction.

## Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

---

Ce README fournit une vue d'ensemble du projet et des instructions pour son utilisation. Pour plus de détails, référez-vous aux commentaires dans le code source.
