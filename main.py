import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

def main():
    # Leitura do arquivo local
    df = pd.read_csv('data.csv', encoding='ISO-8859-1')
    
    print('Dimensão inicial:', df.shape)

    # 1) Remover linhas nulas
    df.dropna(inplace=True)

    # 2) Remover devoluções (InvoiceNo iniciando com 'C')
    df = df[df['InvoiceNo'].str.startswith('C') == False]
    print('Dimensão após limpeza:', df.shape)

    # 3) Criar variável alvo 'Abandoned'
    user_purchase_counts = df.groupby('CustomerID')['InvoiceNo'].nunique()
    def only_purchased_once(customer_id):
        return 1 if user_purchase_counts[customer_id] == 1 else 0
    
    df['Abandoned'] = df['CustomerID'].apply(only_purchased_once)

    # 4) Criar 'TotalPrice'
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

    # 5) Escolher features
    features = df[['Quantity', 'UnitPrice', 'TotalPrice', 'Country']].copy()
    target = df['Abandoned']

    # 6) Codificar 'Country'
    le = LabelEncoder()
    features['Country'] = le.fit_transform(features['Country'])

    # 7) Escalonar
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    y = target

    # 8) Separar treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 9) Definir modelos
    models = {
        'KNN': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(),
        'Naive Bayes': GaussianNB()
    }

    results = []

    # 10) Treinar e avaliar cada modelo
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        # Salvar resultados em uma lista
        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': report['weighted avg']['precision'],
            'Recall': report['weighted avg']['recall'],
            'F1-Score': report['weighted avg']['f1-score']
        })

        print(f\"\\n--- {name} ---\")
        print(\"Accuracy:\", acc)
        print(\"Confusion Matrix:\\n\", cm)
        print(\"Classification Report:\\n\", classification_report(y_test, y_pred))

    # 11) Comparar resultados em DataFrame
    results_df = pd.DataFrame(results)
    print(\"\\n===== Resumo das Métricas =====\")
    print(results_df)

    # 12) Plotar gráfico de barras
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y='Accuracy', data=results_df)
    plt.title('Comparação de Acurácia dos Modelos')
    plt.ylim(0, 1)
    plt.show()

if __name__ == '__main__':
    main()
