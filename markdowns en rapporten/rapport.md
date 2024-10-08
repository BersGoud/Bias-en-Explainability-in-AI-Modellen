# Rapport over Bias en Explainability Analyse in de Sollicitantendataset

## Inleiding
In dit rapport worden de resultaten gepresenteerd van de analyses die zijn uitgevoerd op een dataset van sollicitanten. We onderzoeken zowel de bias in de aannamepercentages op basis van geslacht, leeftijd en opleidingsniveau als de explainability van de voorspellingen van een Random Forest Classifier model. Het doel is om te begrijpen of bepaalde groepen onevenredig worden benadeeld en om inzicht te krijgen in hoe verschillende features bijdragen aan de beslissingen van het model.

## Bias-analyse

### Dataset Inladen
De analyse begint met het inladen van de dataset van sollicitanten. De dataset bevat informatie over verschillende kenmerken van de kandidaten, waaronder geslacht, leeftijd, en opleidingsniveau.

### Bias op basis van Geslacht
Bij de analyse van de bias op basis van geslacht is het percentage aangenomen sollicitanten voor mannen en vrouwen berekend. De resultaten tonen aan dat 85.28% van de vrouwelijke sollicitanten en 83.01% van de mannelijke sollicitanten zijn aangenomen. Een chi-kwadraat toets toont een P-waarde van 0.276, wat aangeeft dat er geen statistisch significante verschillen zijn in de aannamepercentages op basis van geslacht. Dit suggereert dat er geen bias is op basis van geslacht.

### Bias op basis van Leeftijd
Vervolgens zijn de aannamepercentages per leeftijdsgroep geanalyseerd. De resultaten tonen aan dat de aannamepercentages variëren van 65.22% voor de groep 18-25 tot 90.87% voor de groep 36-45. De chi-kwadraat toets voor leeftijdsgroepen heeft een P-waarde van 5.21e-11, wat betekent dat er significante verschillen zijn in de aannamepercentages tussen de verschillende leeftijdsgroepen. Dit wijst op de aanwezigheid van bias op basis van leeftijd.

### Bias op basis van Opleidingsniveau
De analyse van de aannamepercentages op basis van opleidingsniveau toonde aan dat het percentage aangenomen sollicitanten varieert van 81.98% tot 89.58%, afhankelijk van het opleidingsniveau. De chi-kwadraat toets voor opleidingsniveau heeft een P-waarde van 0.553, wat betekent dat er geen statistisch significante verschillen zijn in de aannamepercentages op basis van opleidingsniveau. Dit suggereert dat er geen bias is op basis van opleidingsniveau.

## Explainability-analyse

### Dataset Voorbereiden
In de explainability-analyse wordt de dataset opnieuw ingeladen en voorbereid. Dit omvat het omzetten van categorische kolommen naar numerieke waarden en het splitsen van de dataset in trainings- en testgegevens.

### Trainen van het Model
Een Random Forest Classifier model wordt getraind op de voorbereide data. Dit model wordt gebruikt om de voorspellingen te maken en te analyseren.

### LIME Explainability Toepassen
De explainability-analyse wordt uitgevoerd met behulp van LIME (Local Interpretable Model-agnostic Explanations). Deze techniek helpt om te begrijpen hoe verschillende features bijdragen aan de voorspellingen van het model. Door LIME toe te passen, kunnen we de impact van features zoals `MonthlyIncome` en `Overtime` op de aannamebeslissingen analyseren.

### Belangrijkste Bevindingen
Bij het analyseren van de explainability werden twee features geïdentificeerd die significantere effecten hebben dan de andere:

- **MonthlyIncome**: Een negatieve score van -0.017 geeft aan dat een hoger maandinkomen geassocieerd is met een lagere kans op aanname. Dit suggereert dat er mogelijk bias is tegen kandidaten met een hoger inkomen.

- **Overtime**: Een positieve score van 0.03 geeft aan dat kandidaten die bereid zijn om overuren te werken, een grotere kans hebben om aangenomen te worden. Dit kan leiden tot bias tegen kandidaten die hier niet toe in staat zijn.

### Conclusie
De analyses hebben aangetoond dat:

- Er **geen bias** is op basis van **geslacht**.
- Er **is bias** op basis van **leeftijd**; de aannamepercentages verschillen significant tussen leeftijdsgroepen.
- Er **is bias** op basis van de features **MonthlyIncome** (-0.017) en **Overtime** (0.03) in de explainability-analyse.

### Aanbevelingen
Het is belangrijk om het wervingsproces te herzien, vooral met betrekking tot de leeftijdsgroepen en de invloed van inkomen en overuren. Training voor wervingsmanagers over onbewuste bias en het implementeren van eerlijke criteria kan bijdragen aan een inclusiever beleid. Daarnaast is het aanbevelenswaardig om de beslissingen van het wervingsmodel verder te onderzoeken en ervoor te zorgen dat alle kandidaten gelijke kansen krijgen.