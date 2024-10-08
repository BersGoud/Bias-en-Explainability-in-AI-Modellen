from scipy.stats import chi2_contingency
import pandas as pd
# Laad de dataset
data = pd.read_csv('HR_Analytics.csv')
# Bekijk de eerste paar rijen van de dataset
print(data.head())

# Bereken het percentage 'Attrition' per geslacht
gender_attrition = data.groupby('Gender')['Attrition'].value_counts(
    normalize=True).unstack() * 100
# Bekijk het percentage van sollicitanten die 'No' (niet vertrokken) zijn
gender_attrition_no = gender_attrition['No']
print(gender_attrition_no)

# Maak een kruistabel van Gender en Attrition
contingency_table = pd.crosstab(data['Gender'], data['Attrition'])
# Voer de chi-kwadraat toets uit
chi2, p, dof, expected = chi2_contingency(contingency_table)
# Resultaten afdrukken
print("Chi2-statistiek:", chi2)
print("P-waarde:", p)
print("Aantal vrijheidsgraden:", dof)
print("Verwachte frequenties:\n", expected)

# Maak categorieën voor leeftijdsgroepen
data['AgeGroup'] = pd.cut(data['Age'], bins=[18, 25, 35, 45, 55, 65], labels=[
                          '18-25', '26-35', '36-45', '46-55', '56-65'])
# Bereken het percentage 'Attrition' (No) per leeftijdsgroep
age_attrition = data.groupby('AgeGroup')['Attrition'].value_counts(
    normalize=True).unstack() * 100
# Toon het percentage van sollicitanten die zijn aangenomen per leeftijdsgroep
age_attrition_no = age_attrition['No']
print(age_attrition_no)

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

# Bereken het percentage 'Attrition' (No) per opleidingsniveau
education_attrition = data.groupby(
    'Education')['Attrition'].value_counts(normalize=True).unstack() * 100
# Toon het percentage van sollicitanten die zijn aangenomen per opleidingsniveau
education_attrition_no = education_attrition['No']
print(education_attrition_no)

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
