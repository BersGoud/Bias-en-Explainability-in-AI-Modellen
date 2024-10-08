Hier is het `explainability_analyse.md` bestand met dezelfde structuur, flow en verduidelijkingswijze als het `bias_analyse.md` bestand:

---

# Explainability-analyse in Sollicitantendataset

## Inleiding
In deze analyse onderzoeken we de explainability van een dataset van sollicitanten op basis van een Random Forest Classifier model. We willen begrijpen hoe verschillende features bijdragen aan de voorspelling van de uitkomst, namelijk of een sollicitant wordt aangenomen of niet (Attrition).

## 1. Dataset Inladen
We beginnen met het inladen van de benodigde bibliotheken en de dataset.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import lime
import lime.lime_tabular

# Laad de dataset
data = pd.read_csv('HR_Analytics.csv')
# Bekijk de eerste paar rijen van de dataset
print(data.head())
```

## 2. Voorbereiden van de Data
We bereiden de dataset voor door de categorische kolommen om te zetten naar numerieke waarden en het verwijderen van eventuele missende waarden.

```python
# Converteer categorische kolommen naar numeriek
label_encoder = LabelEncoder()
categorical_columns = ['BusinessTravel', 'Department', 'EducationField', 'Gender',
                       'JobRole', 'MaritalStatus', 'Over18', 'OverTime', 'AgeGroup', 'SalarySlab']

for col in categorical_columns:
    data[col] = label_encoder.fit_transform(data[col])

# Label voor de target variable 'Attrition'
data['Attrition'] = label_encoder.fit_transform(data['Attrition'])

# Selecteer features en target
X = data.drop(columns=['Attrition', 'EmpID'])
y = data['Attrition']

# Verwijderen van missing values
X = X.dropna()
y = y[X.index]

# Splitsen in training en test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## 3. Trainen van het Model
We trainen een Random Forest Classifier model op de voorbereide data.

```python
# Trainen van het model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

## 4. LIME Explainability Toepassen
### 4.1 Instantieer de LIME Explainability Tool
We gebruiken LIME om de beslissingen van het model te verklaren en om te begrijpen welke kenmerken de meeste invloed hebben op de voorspellingen.

```python
from lime.lime_tabular import LimeTabularExplainer

# Instantieer LIME explainability tool
explainer = LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X.columns,
    class_names=['No Attrition', 'Attrition'],
    mode='classification',
    discretize_continuous=False
)
```

### 4.2 Kies een Willekeurige Sample uit de Testset
We kiezen een willekeurige sample uit de testset om de explainability te analyseren.

```python
# Kies een willekeurige sample uit de testset
i = random.choice(range(len(X_test)))  # Je kunt deze index aanpassen voor andere rijen
exp = explainer.explain_instance(
    data_row=X_test.iloc[i],
    predict_fn=model.predict_proba  # Functie om voorspellingen te genereren
)

# Toon de uitleg in een Jupyter Notebook of exporteer naar een HTML-bestand
exp.show_in_notebook(show_all=False)
exp.save_to_file('lime_explanation_single.html')
```

### output
De LIME-output toont de impact van verschillende features op de voorspelling voor de gekozen instance. Het geeft aan welke kenmerken de grootste positieve of negatieve invloed hebben op de voorspelling (bijvoorbeeld "Attrition" of "No Attrition").

## 5. Meerdere Instances Analyseren
We willen de explainability voor meerdere instances analyseren en de gemiddelde effecten van elke feature berekenen.

```python
# Functie voor LIME explainability met meerdere instances
def run_lime_explainability(X_train, X_test, model, instance_count):
    explanations = []
    feature_effects = {feature: 0 for feature in X.columns}

    random_indices = random.sample(range(len(X_test)), instance_count)

    for i in random_indices:
        exp = explainer.explain_instance(
            data_row=X_test.iloc[i],
            predict_fn=model.predict_proba
        )
        explanations.append({
            'Instance': i,
            'Explanations': exp.as_list()  # Lijst van kenmerken en hun effect
        })

        # Cumulatieve effecten bijwerken
        for feature, effect in exp.as_list():
            feature_effects[feature] += effect

    # Gemiddelden berekenen
    average_effects = {
        feature: effect / instance_count for feature, effect in feature_effects.items()
    }

    return explanations, average_effects
```

### output
Na het uitvoeren van deze functie, kun je de resultaten opslaan in een CSV-bestand voor verdere analyse.

```python
# Opslaan van gemiddelde effecten in CSV
with open('lime_average_effects.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Feature', 'Average Effect'])  # Headers
    for feature, avg_effect in average_effects.items():
        writer.writerow([feature, avg_effect])

print("Average effects saved to lime_average_effects.csv")
```

## Conclusie
De explainability-analyse heeft ons in staat gesteld om te begrijpen hoe verschillende features bijdragen aan de voorspellingen van het model. Dit helpt bij het identificeren van features die mogelijk bias kunnen vertonen en bij het verbeteren van de transparantie van ons AI-model.

### MonthlyIncome & OverTime
Er zijn 2 features die meer invloed uiten dan de rest. MonthlyIncome heeft een negatieve score van -0.017 en OverTime heeft een positieve score van 0.03. 

De andere features hebben een absolute invloed van onder de 0.01. En zijn te klein om van patronen te kunnen spreken.

#### MonthlyIncome
Wanneer de explainability-analyse aangeeft dat de feature MonthlyIncome een effect heeft van -0.017, betekent dit dat er een negatieve relatie is tussen het maandinkomen en de voorspelling van de uitkomst (in dit geval, of een sollicitant wordt aangenomen of niet).

Een negatieve score van -0.017 voor MonthlyIncome geeft aan dat als het maandinkomen toeneemt, de kans op de uitkomst afneemt. Dit suggereert dat hogere maandinkomens geassocieerd kunnen zijn met een lagere kans op aanname.

De waarde van -0.017 is relatief klein, maar het geeft aan dat er een consistent patroon is in de manier waarop het model de waarde van MonthlyIncome gebruikt om zijn voorspellingen te maken. Het betekent dat elke eenheidstoename in het maandinkomen een vermindering van 0.017 veroorzaakt in de kans op aanname.

#### OverTime

Overtime heeft een positieve score van 0.03. Dit betekent dat als een sollicitant bereid is om overuren te werken, de kans op aanname toeneemt.

Deze positieve associatie kan verschillende oorzaken hebben. 
In sommige bedrijven of sectoren wordt de bereidheid om overuren te maken gezien als een teken van inzet en toewijding, wat kan leiden tot een hogere kans op aanname.
Werken met overuren kan ook worden geïnterpreteerd als een teken dat de kandidaat beter in staat is om zich aan te passen aan de behoeften van het bedrijf, vooral in tijden van hoge werkdruk of deadlines.
Voor sommige functies kan het noodzakelijk zijn om overuren te maken, en kandidaten die dit kunnen of willen doen, worden mogelijk als geschikter beschouwd.

De positieve score voor Overtime kan wijzen op een voorkeur in het wervingsproces voor kandidaten die bereid zijn overuren te maken, wat kan resulteren in een onevenredige behandeling van kandidaten die om legitieme redenen geen overuren kunnen of willen maken.

Het kan nuttig zijn om te overwegen hoe dit effect kan worden aangepakt. Bijvoorbeeld:

Het verbeteren van de werk-privébalans binnen het bedrijf en het bieden van alternatieve manieren om inzet en betrokkenheid te tonen zonder dat overuren vereist zijn.

Het ontwikkelen van een transparant beleid rond werkuren en verwachtingen om ervoor te zorgen dat alle kandidaten gelijke kansen krijgen, ongeacht hun bereidheid om overuren te maken.

Dit benadrukt de noodzaak om de beslissingen van het wervingsmodel verder te onderzoeken. Het is essentieel om te waarborgen dat het aannemen van kandidaten eerlijk en transparant gebeurt, zonder onterecht voorkeur te geven aan degenen die overuren kunnen of willen maken.



---
