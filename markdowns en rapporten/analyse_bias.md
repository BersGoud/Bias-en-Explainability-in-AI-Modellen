# Bias-analyse in Sollicitantendataset

## Inleiding
In deze analyse onderzoeken we bias in een dataset van sollicitanten op basis van geslacht, leeftijd en opleidingsniveau. We willen begrijpen of er significante verschillen zijn in de aannamepercentages tussen verschillende groepen.

## 1. Dataset Inladen
We beginnen met het inladen van de benodigde bibliotheken en de dataset.

```python
import pandas as pd
# Laad de dataset
data = pd.read_csv('HR_Analytics.csv')
# Bekijk de eerste paar rijen van de dataset
print(data.head())
```
## 2. Bias op basis van Geslacht
We analyseren eerst de bias op basis van geslacht.

### 2.1 Acceptatiepercentages per Geslacht
We berekenen het percentage aangenomen sollicitanten (Attrition = 'No') 
voor mannen en vrouwen.

We groeperen de data op basis van geslacht en berekenen de percentages voor elke waarde van de 'Attrition'-kolom.
```python
# Bereken het percentage 'Attrition' per geslacht
gender_attrition = data.groupby('Gender')['Attrition'].value_counts(normalize=True).unstack() * 100
# Bekijk het percentage van sollicitanten die 'No' (niet vertrokken) zijn
gender_attrition_no = gender_attrition['No']
print(gender_attrition_no)
```

### output
  Gender | Attrition
- Female | 85.279188
- Male   | 83.014623

### 2.2 Chi-Kwadraat Toets voor Geslacht
De chi-kwadraat toets is een statistische methode die wordt gebruikt om de onafhankelijkheid tussen twee categorische variabelen te onderzoeken door de geobserveerde en verwachte frequenties te vergelijken. In onze analyse gebruiken we deze toets om te identificeren of er significante bias aanwezig is in de aannamebeslissingen van sollicitanten op basis van kenmerken zoals geslacht, leeftijd of opleidingsniveau. Dit helpt ons te bepalen of bepaalde groepen onevenredig worden benadeeld in het wervingsproces. Door de resultaten van de chi-kwadraat toets te evalueren, kunnen we aanbevelingen doen voor een eerlijker AI-model.

We voeren een chi-kwadraat toets uit om te bepalen of het verschil in aannamepercentages tussen mannen en vrouwen statistisch significant is.

We maken een kruistabel van geslacht en aanname en voeren de chi-kwadraat toets uit om de significantie te testen.
```python
from scipy.stats import chi2_contingency
# Maak een kruistabel van Gender en Attrition
contingency_table = pd.crosstab(data['Gender'], data['Attrition'])
# Voer de chi-kwadraat toets uit
chi2, p, dof, expected = chi2_contingency(contingency_table)
# Resultaten afdrukken
print("Chi2-statistiek:", chi2)
print("P-waarde:", p)
print("Aantal vrijheidsgraden:", dof)
print("Verwachte frequenties:\n", expected)
```
### output
- Chi2-statistiek: 1.1864423346425068
- P-waarde: 0.27604835591828547
- Aantal vrijheidsgraden: 1
- Verwachte frequenties:
    - [[495.96081081 | 95.03918919]
    - [746.03918919  | 142.96081081]]

### besluit
De P-waarde is groter dan 0.05, wat betekent dat we de nulhypothese niet kunnen verwerpen. Dit suggereert dat er geen statistisch significante verschillen zijn in de aannamepercentages tussen mannen en vrouwen. Er lijkt geen bias op basis van geslacht te zijn.

## 3. Bias op basis van Leeftijd
### 3.1 Leeftijdscategorieën maken en Acceptatiepercentages Berekenen
We maken leeftijdscategorieën en berekenen de aannamepercentages per leeftijdsgroep.

We gebruiken pd.cut om leeftijdscategorieën te maken en vervolgens berekenen we het percentage aangenomen sollicitanten voor elke groep.

```python
# Maak categorieën voor leeftijdsgroepen
data['AgeGroup'] = pd.cut(data['Age'], bins=[18, 25, 35, 45, 55, 65], labels=['18-25', '26-35', '36-45', '46-55', '56-65'])
# Bereken het percentage 'Attrition' (No) per leeftijdsgroep
age_attrition = data.groupby('AgeGroup')['Attrition'].value_counts(normalize=True).unstack() * 100
# Toon het percentage van sollicitanten die zijn aangenomen per leeftijdsgroep
age_attrition_no = age_attrition['No']
print(age_attrition_no)
```
### output
AgeGroup| Attrition
- 18-25 | 65.217391
- 26-35 | 81.014730
- 36-45 | 90.870488
- 46-55 | 88.157895
- 56-65 | 82.978723

## 3.2 Chi-Kwadraat Toets voor Leeftijd
We maken een kruistabel voor leeftijdsgroepen en voeren de chi-kwadraat toets uit om de significantie te testen.

```python
# Maak een kruistabel van AgeGroup en Attrition
age_contingency_table = pd.crosstab(data['AgeGroup'], data['Attrition'])
# Chi-kwadraat toets uitvoeren
chi2_age, p_age, dof_age, expected_age = chi2_contingency(
    age_contingency_table)

# Resultaten afdrukken
print("Chi2-statistiek:", chi2_age)
print("P-waarde:", p_age)
print("Aantal vrijheidsgraden:", dof_age)
print("Verwachte frequenties:\n", expected_age)
```
### output
- Chi2-statistiek: 54.02036481968599
- P-waarde: 5.211260237202568e-11
- Aantal vrijheidsgraden: 4
- Verwachte frequenties:
  - [[ 96.71875    | 18.28125   ]
  - [513.87092391  | 97.12907609]
  - [396.1263587   | 74.8736413 ]
  - [191.75543478  | 36.24456522]
  - [ 39.52853261  | 7.47146739]]

### besluit
De P-waarde is veel kleiner dan 0.05, wat betekent dat we de nulhypothese verwerpen. Dit suggereert dat er significante verschillen zijn in de aannamepercentages tussen verschillende leeftijdsgroepen. Er lijkt bias aanwezig te zijn op basis van leeftijd.

# 4. Bias op basis van Opleidingsniveau
## 4.1 Acceptatiepercentages per Opleidingsniveau
We analyseren het percentage aangenomen sollicitanten op basis van opleidingsniveau.

### Uitleg:
We groeperen de data op basis van opleidingsniveau en berekenen de percentages voor elke waarde van de 'Attrition'-kolom.

```python
# Bereken het percentage 'Attrition' (No) per opleidingsniveau
education_attrition = data.groupby('Education')['Attrition'].value_counts(normalize=True).unstack() * 100
# Toon het percentage van sollicitanten die zijn aangenomen per opleidingsniveau
education_attrition_no = education_attrition['No']
print(education_attrition_no)
```
### output
Education | Attrition
- 1 | 81.976744
- 2 | 84.452297
- 3 | 82.698962
- 4 | 85.463659
- 5 | 89.583333

## 4.2 Chi-Kwadraat Toets voor Opleidingsniveau
We voeren een chi-kwadraat toets uit om te bepalen of de verschillen in aannamepercentages op basis van opleidingsniveau significant zijn.

### Uitleg:
We maken een kruistabel voor opleidingsniveau en voeren de chi-kwadraat toets uit om de significantie te testen.
```python
# Maak een kruistabel van Education en Attrition
education_contingency_table = pd.crosstab(data['Education'], data['Attrition'])
# Chi-kwadraat toets uitvoeren
chi2_education, p_education, dof_education, expected_education = chi2_contingency(
    education_contingency_table)

# Resultaten afdrukken
print("Chi2-statistiek (Opleiding):", chi2_education)
print("P-waarde (Opleiding):", p_education)
print("Aantal vrijheidsgraden (Opleiding):", dof_education)
print("Verwachte frequenties (Opleiding):\n", expected_education)
```
### output
- Chi2-statistiek (Opleiding): 3.0246218569914087
- P-waarde (Opleiding): 0.5537134799192043
- Aantal vrijheidsgraden (Opleiding): 4
- Verwachte frequenties (Opleiding):
  - [[144.34054054 |  27.65945946]
  - [237.49054054  | 45.50945946]
  - [485.05135135  | 92.94864865]
  - [334.83648649  | 64.16351351]
  - [ 40.28108108  | 7.71891892]]

### besluit
De P-waarde is groter dan 0.05, wat betekent dat we de nulhypothese niet kunnen verwerpen. Dit suggereert dat er geen statistisch significante verschillen zijn in de aannamepercentages op basis van opleidingsniveau. Er lijkt geen bias te zijn op basis van opleidingsniveau.

## Conclusie
Op basis van de analyses concluderen we dat:

- Er **geen bias** is op basis van **geslacht**.
- Er **is bias** op basis van **leeftijd**; de aannamepercentages verschillen significant tussen leeftijdsgroepen.
- Er **geen bias** is op basis van **opleidingsniveau**.

### Aanbevelingen
Gezien de bevindingen over bias op basis van leeftijd, adviseren we om het wervingsproces te herzien en te onderzoeken waarom bepaalde leeftijdsgroepen mogelijk benadeeld worden. Dit kan helpen om een eerlijker en inclusiever wervingsbeleid te ontwikkelen.