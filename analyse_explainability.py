import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import lime
import lime.lime_tabular
import argparse
import random
import csv
import json

# Functie om de data voor te bereiden


def prepare_data():
    # Laad de dataset
    data = pd.read_csv('HR_Analytics.csv')

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

    # Verwijderen van missing values
    X = X.dropna()
    # Zorg ervoor dat de labels passen bij de nieuwe X
    y = data.loc[X.index, 'Attrition']

    # Splitsen in training en test data
    return train_test_split(X, y, test_size=0.2, random_state=42), X.columns

# Functie voor LIME explainability


def run_lime_explainability(X_train, X_test, model, instance_count, output_format, feature_names):
    # Instantieer LIME explainability tool
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=feature_names,
        class_names=['No Attrition', 'Attrition'],
        mode='classification',
        discretize_continuous=False
    )

    explanations = []  # Lijst voor alle explanations bij meerdere instanties

    if instance_count == 1:
        i = random.choice(range(len(X_test)))  # Kies een willekeurige instance
        exp = explainer.explain_instance(
            data_row=X_test.iloc[i],
            predict_fn=model.predict_proba
        )

        # Genereer en sla de HTML-output op
        html_output = exp.as_html()
        with open("output_explainability/html/lime_explanation_single", "w", encoding='utf-8') as f:
            f.write(html_output)

        print("Explanation for instance saved to lime_explanation_single.html")
        # Ook tonen in notebook als dat gewenst is
        exp.show_in_notebook(show_all=False)

    else:
        random_indices = random.sample(range(len(X_test)), instance_count)

        # Voor het opslaan van cumulatieve effecten
        feature_effects = {feature: 0 for feature in feature_names}

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

        # Genereer HTML-output
        html_output = "<html><head><title>LIME Explanations</title></head><body>"
        html_output += "<h1>Combined LIME Explanations</h1>"

        for explanation in explanations:
            html_output += f"<h2>Explanation for instance {explanation['Instance']}</h2>"
            html_output += "<ul>"
            for feature, effect in explanation['Explanations']:
                html_output += f"<li>{feature}: {effect}</li>"
            html_output += "</ul>"

        html_output += "</body></html>"

        with open("output_explainability/html/lime_explanation_combined.html", "w", encoding='utf-8') as f:
            f.write(html_output)

        print("Combined explanations saved to lime_explanation_combined.html")

        # Opslaan als CSV
        with open('output_explainability/csv/lime_explanation_summary.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Instance', 'Feature', 'Effect'])  # Headers

            for explanation in explanations:
                for feature, effect in explanation['Explanations']:
                    writer.writerow(
                        [explanation['Instance'], feature, effect]
                    )

        print("Combined explanations saved to lime_explanation_summary.csv")

        # Opslaan als JSON
        with open('output_explainability/lime_explanation_summary.json', 'w', encoding='utf-8') as file:
            json.dump(explanations, file)
        print("Combined explanations saved to lime_explanation_summary.json")

        # Gemiddelden opslaan in CSV
        with open('output_explainability/csv/lime_average_effects.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Feature', 'Average Effect'])  # Headers
            for feature, avg_effect in average_effects.items():
                writer.writerow([feature, avg_effect])

        print("Average effects saved to lime_average_effects.csv")

# Hoofdfunctie


def main():
    parser = argparse.ArgumentParser(
        description='LIME Explainability Analysis')
    parser.add_argument('--instances', type=int, choices=[1, 10, 20], default=1,
                        help='Number of instances to analyze (1, 10, or 20).')
    parser.add_argument('--format', type=str, choices=['html', 'csv', 'json', 'notebook'], default='html',
                        help='Output format (html, csv, json, or notebook).')
    args = parser.parse_args()

    # Voorbereiden van data
    (X_train, X_test, y_train, y_test), feature_names = prepare_data()
    # Trainen van het model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Voer de LIME explainability uit
    run_lime_explainability(X_train, X_test, model,
                            args.instances, args.format, feature_names)


if __name__ == '__main__':
    main()
