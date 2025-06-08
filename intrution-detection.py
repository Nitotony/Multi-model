import pandas as pd
from joblib import load
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import os
import numpy as np

columns = ['Flow Duration', 'Total Fwd Packet', 'Total Bwd packets', 'Total Length of Fwd Packet',
           'Total Length of Bwd Packet', 'Fwd Packet Length Max', 'Fwd Packet Length Min', 
           'Fwd Packet Length Mean', 'Fwd Packet Length Std', 'Bwd Packet Length Max',
           'Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Bwd Packet Length Std',
           'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max',
           'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 
           'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 
           'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 
           'Fwd Header Length', 'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s', 
           'Packet Length Min', 'Packet Length Max', 'Packet Length Mean', 'Packet Length Std', 
           'Packet Length Variance', 'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 
           'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count', 'CWR Flag Count', 'ECE Flag Count', 
           'Down/Up Ratio', 'Average Packet Size', 'Fwd Segment Size Avg', 'Bwd Segment Size Avg', 
           'Fwd Bytes/Bulk Avg', 'Fwd Packet/Bulk Avg', 'Fwd Bulk Rate Avg', 'Bwd Bytes/Bulk Avg',
           'Bwd Packet/Bulk Avg', 'Bwd Bulk Rate Avg', 'Subflow Fwd Packets', 'Subflow Fwd Bytes',
           'Subflow Bwd Packets', 'Subflow Bwd Bytes', 'FWD Init Win Bytes', 'Bwd Init Win Bytes',
           'Fwd Act Data Pkts', 'Fwd Seg Size Min', 'Active Mean', 'Active Std', 'Active Max',
           'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min']

attack_map = {
    0: "Benign",

    3: "DoS",
    4: "Exploits",
    5: "Fuzzers",

    7: "Reconnaissance",
}

rows = int(input("Enter row range:"))
data = pd.read_csv("ip_data.csv")

simulation_data = data.head(rows)

simulation_data_filtered = simulation_data[columns]


binary_clf = load("binary_classifier_rf.pkl")


simulation_data_filtered['binary_label'] = binary_clf.predict(simulation_data_filtered)


non_benign_data = simulation_data_filtered[simulation_data_filtered['binary_label'] == 1]


scaler = load("scaler.pkl")
label_encoder = load("label_encoder.pkl")


non_benign_data_scaled = scaler.transform(non_benign_data[columns])


print("Select a multiclass model to use for prediction:")
print("1: Random Forest")
print("2: LightGBM")
print("3: Neural Network")

choice = int(input("Enter your choice (1/2/3): "))

if choice == 1:
    multi_clf = load("random_forest_model.pkl")
    print("Using Random Forest model for multiclass predictions.")
elif choice == 2:
    multi_clf = load("lgb_classifier_model.pkl")
    print("Using LightGBM model for multiclass predictions.")
elif choice == 3:
    from tensorflow.keras.models import load_model
    multi_clf = load_model("multiclass_nn_model.h5")
    print("Using Neural Network model for multiclass predictions.")
else:
    raise ValueError("Invalid choice. Please select 1, 2, or 3.")

if choice == 3:  
    non_benign_predictions = multi_clf.predict(non_benign_data_scaled)
    non_benign_predictions = non_benign_predictions.argmax(axis=1)
else:
    non_benign_predictions = multi_clf.predict(non_benign_data_scaled)


non_benign_data['multiclass_label'] = label_encoder.inverse_transform(non_benign_predictions)


simulation_data_with_predictions = simulation_data.copy()


simulation_data_with_predictions['Prediction'] = "Unknown"


benign_indices = simulation_data_filtered[simulation_data_filtered['binary_label'] == 0].index
simulation_data_with_predictions.loc[benign_indices, 'Prediction'] = "Benign"


non_benign_indices = non_benign_data.index
non_benign_data['multiclass_label'] = non_benign_data['multiclass_label'].map(attack_map)
simulation_data_with_predictions.loc[non_benign_indices, 'Prediction'] = non_benign_data['multiclass_label']


result_df = simulation_data_with_predictions


print(result_df.head(10))

output_dir = "network_analysis_plots"
os.makedirs(output_dir, exist_ok=True)


def plot_benign_vs_attack(df, output_dir):
    benign_count = df["Prediction"].value_counts().get("Benign", 0)
    attack_count = len(df) - benign_count  

    labels = ['Benign', 'Attack']
    sizes = [benign_count, attack_count]
    colors = ['#66c2a5', '#fc8d62']
    
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, wedgeprops={'edgecolor': 'black'})
    ax.set_title("Benign vs Attack Types", fontsize=16)
    
    pie_chart_path = os.path.join(output_dir, "benign_vs_attack_pie_chart.png")
    plt.savefig(pie_chart_path)
    plt.close()
    return pie_chart_path


def plot_attack_types(df, output_dir):
    attack_type_counts = df[df["Prediction"] != "Benign"]["Prediction"].value_counts()

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(attack_type_counts, labels=attack_type_counts.index, autopct='%1.1f%%', startangle=90, wedgeprops={'edgecolor': 'black'})
    ax.set_title("Attack Types Breakdown", fontsize=16)
    
    attack_pie_chart_path = os.path.join(output_dir, "attack_types_pie_chart.png")
    plt.savefig(attack_pie_chart_path)
    plt.close()
    return attack_pie_chart_path


def plot_top_ips(df, output_dir):
    src_counts = df["Src IP"].value_counts().head(10)
    dst_counts = df["Dst IP"].value_counts().head(10)

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    sns.barplot(x=src_counts.values, y=src_counts.index, ax=ax[0], palette="viridis")
    ax[0].set_title("Top 10 Source IPs by Frequency")
    ax[0].set_xlabel("Frequency")
    ax[0].set_ylabel("Src IP")

    sns.barplot(x=dst_counts.values, y=dst_counts.index, ax=ax[1], palette="magma")
    ax[1].set_title("Top 10 Destination IPs by Frequency")
    ax[1].set_xlabel("Frequency")
    ax[1].set_ylabel("Dst IP")

    chart_path = os.path.join(output_dir, "bar_chart_src_dst.png")
    plt.tight_layout()
    plt.savefig(chart_path)
    plt.close()
    return chart_path


def plot_prediction_distribution(df, output_dir):
    prediction_counts = df["Prediction"].value_counts()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(prediction_counts.index, prediction_counts.values, color=["#66c2a5", "#fc8d62"])
    ax.set_title("Prediction Distribution", fontsize=16)
    ax.set_xlabel("Predictions", fontsize=14)
    ax.set_ylabel("Count", fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    bar_chart_path = os.path.join(output_dir, "bar_chart_predictions.png")
    plt.savefig(bar_chart_path)
    plt.close()
    return bar_chart_path



def plot_correlation_heatmap(df, output_dir):
    numerical_columns = [
        "Flow Duration", "Fwd Packet Length Max", "Fwd Packet Length Mean", 
        "Fwd Packet Length Std", "Bwd Packet Length Max", "Flow Bytes/s", 
        "Flow Packets/s", "Fwd IAT Total", "Fwd Packets/s", "Bwd Packets/s", 
        "Packet Length Min", "Packet Length Max", "Packet Length Mean", 
        "Packet Length Std", "Down/Up Ratio", "Average Packet Size", 
        "Fwd Segment Size Avg"
    ]
    corr_matrix = df[numerical_columns].corr()
    print(corr_matrix)
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title("Correlation Heatmap", fontsize=16)

    heatmap_path = os.path.join(output_dir, "correlation_heatmap.png")
    plt.tight_layout()
    plt.savefig(heatmap_path)
    plt.close()
    return heatmap_path


def plot_feature_distribution(df, output_dir):
    numerical_columns = ["Flow Duration", "Flow Bytes/s", "Flow Packets/s", "Average Packet Size"]
    fig, ax = plt.subplots(2, 2, figsize=(14, 12))
    ax = ax.flatten()

    for i, column in enumerate(numerical_columns):
        sns.histplot(df[column].dropna(), kde=True, ax=ax[i], bins=30, color="blue")
        ax[i].set_title(f"Distribution of {column}", fontsize=14)

    dist_path = os.path.join(output_dir, "feature_distribution.png")
    plt.tight_layout()
    plt.savefig(dist_path)
    plt.close()
    return dist_path


def plot_boxplots(df, output_dir):
    numerical_columns = ["Flow Duration", "Flow Bytes/s", "Average Packet Size"]
    fig, ax = plt.subplots(1, len(numerical_columns), figsize=(16, 6))

    for i, column in enumerate(numerical_columns):
        sns.boxplot(data=df, y=column, ax=ax[i], palette="coolwarm")
        ax[i].set_title(f"Boxplot: {column}")

    boxplot_path = os.path.join(output_dir, "boxplots.png")
    plt.tight_layout()
    plt.savefig(boxplot_path)
    plt.close()
    return boxplot_path


def plot_pairplot(df, output_dir):
    selected_columns = ["Flow Duration", "Flow Bytes/s", "Flow Packets/s", "Average Packet Size"]
    sns.pairplot(df[selected_columns], diag_kind="kde", palette="husl")

    pairplot_path = os.path.join(output_dir, "pairplot.png")
    plt.savefig(pairplot_path)
    plt.close()
    return pairplot_path


def plot_cdf(df, output_dir):
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    for i, column in enumerate(["Flow Duration", "Flow Bytes/s"]):
        sorted_data = np.sort(df[column].dropna())
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        ax[i].plot(sorted_data, cdf, marker=".", linestyle="none", color="blue")
        ax[i].set_title(f"CDF of {column}")
        ax[i].set_xlabel(column)
        ax[i].set_ylabel("CDF")

    cdf_path = os.path.join(output_dir, "cdf_plots.png")
    plt.tight_layout()
    plt.savefig(cdf_path)
    plt.close()
    return cdf_path


bar_chart_path = plot_top_ips(result_df, output_dir)
pie_chart_path = plot_prediction_distribution(result_df, output_dir)
# network_graph_path = plot_network_graph(result_df, output_dir)
heatmap_path = plot_correlation_heatmap(result_df, output_dir)
dist_path = plot_feature_distribution(result_df, output_dir)
boxplot_path = plot_boxplots(result_df, output_dir)
pairplot_path = plot_pairplot(result_df, output_dir)
cdf_path = plot_cdf(result_df, output_dir)
benign_vs_attack_pie_path = plot_benign_vs_attack(result_df, output_dir)
attack_types_pie_path = plot_attack_types(result_df, output_dir)


html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Network Analysis Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            text-align: center;
            margin: 0 auto;
            max-width: 800px;
        }}
        h1 {{
            color: #4CAF50;
        }}
        h2 {{
            color: #555;
        }}
        img {{
            border: 2px solid #ddd;
            border-radius: 4px;
            padding: 5px;
            width: 80%;
            margin: 20px auto;
        }}
    </style>
</head>
<body>
    <h1>Network Analysis Report</h1>
    
    <h2>1. Benign vs Attack Types</h2>
    <img src="{benign_vs_attack_pie_path}" alt="Benign vs Attack Types Pie Chart" width="800px">
    
    <h2>2. Attack Types Breakdown</h2>
    <img src="{attack_types_pie_path}" alt="Attack Types Breakdown Pie Chart" width="800px">
    
    <h2>3. Top Src IPs and Dst IPs by Frequency</h2>
    <img src="{bar_chart_path}" alt="Bar Chart" width="800px">
    
    <h2>4. Prediction Distribution</h2>
    <img src="{pie_chart_path}" alt="Pie Chart" width="800px">
  
    <h2>5. Correlation Heatmap</h2>
    <img src="{heatmap_path}" alt="Heatmap" width="800px">
    
    <h2>6. Distribution of Key Features</h2>
    <img src="{dist_path}" alt="Feature Distributions" width="800px">

    <h2>7. Boxplots for Key Numerical Features</h2>
    <img src="{boxplot_path}" alt="Boxplots">
    
    <h2>8. Pairplot for Selected Features</h2>
    <img src="{pairplot_path}" alt="Pairplot">
    
    <h2>9. Cumulative Distribution Function (CDF)</h2>
    <img src="{cdf_path}" alt="CDF Plots">
</body>
</html>
"""

# Save the report
report_path = "network_analysis_report.html"
with open(report_path, "w") as f:
    f.write(html_template)

print(f"Report saved to {report_path}")


from sqlalchemy import create_engine

def save_to_mysql(df, host, port, user, password, database, table_name):
    connection_string = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
    engine = create_engine(connection_string)
    
    df.to_sql(table_name, con=engine, if_exists="replace", index=False)

    print(f"DataFrame successfully saved to MySQL table: {table_name}")

host = 'localhost'  
port = '3306'             
user = 'root'  
password = 'ntc4344718'
database = 'network'
table_name = 'network_analysis_results'


print("Saving to DB..")
filtered_df = result_df.drop(columns=['Label'], errors='ignore')

filtered_df = filtered_df[filtered_df['Prediction'] != 'Benign']



print(filtered_df.head(30))
save_to_mysql(filtered_df, host, port, user, password, database, table_name)
