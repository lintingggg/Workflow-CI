import pandas as pd
from sklearn.neighbors import NearestNeighbors
import mlflow
import mlflow.sklearn

# Main function / Modelling
def main():
    print("=== Mulai Membaca Dataset ===")

    df = pd.read_csv('movie_feelings_dataset_preprocessing.csv')

    kolom_bukan_fitur = ['imdb_id','title_year']
    features = df.drop(columns=kolom_bukan_fitur, errors='ignore')
    features = features.select_dtypes(include=['number'])

    print("=== Memulai Training CI/CD ===")
    with mlflow.start_run():
        k = 3
        matrix = 'cosine'

        mlflow.log_param(f"n_neighbours {k}")
        mlflow.log_param(f"metric {matrix}")

        print(f"Melatih model KNN dengan K {k} dan metric {matrix}")
        model = NearestNeighbors(n_neighbors=k, metric=matrix)
        model.fit(features)

        distances = model.kneighbors(features)
        mean_distances = distances.mean()

        mlflow.log_metric(f"Rata-rata jarak {mean_distances}")
        mlflow.sklearn.log_model(f"{model} model")
        print("=== Training selesai dan model disimpan ===")

if __name__ == "__main__":
    main()