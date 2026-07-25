import os
import pickle
import pandas as pd
import kagglehub

from sklearn.datasets import fetch_openml
from ucimlrepo import fetch_ucirepo

from sbtab.data.schema import TabularSchema

def prepare_smart_df(df, target_col, task_type):
    df = df.copy()
    schema = TabularSchema.infer_from_dataframe(df, target_col=target_col)

    df.attrs['target_variable'] = target_col
    df.attrs['task_type'] = task_type
    df.attrs['features'] = schema.feature_cols
    df.attrs['feature_types'] = {
        "continuous": schema.continuous_cols,
        "discrete": schema.discrete_cols,
        "categorical": schema.categorical_cols
    }
    return df

def save_pickle(data_dict, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data_dict, f)
    print(f"File {file_path} is successfully saved.")

def load_kaggle_csv(handle):
    """Downloads a Kaggle dataset and returns the main CSV as a DataFrame."""
    path = kagglehub.dataset_download(handle)
    csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError(f"No CSV found in {path}")

    csv_files.sort(key=lambda x: os.path.getsize(os.path.join(path, x)), reverse=True)
    csv_path = os.path.join(path, csv_files[0])

    with open(csv_path, 'r', encoding='utf-8') as f:
        first_line = f.readline()
        sep = ';' if ';' in first_line and ',' not in first_line else ','

    return pd.read_csv(csv_path, sep=sep)

def main():
    TARGET_COL_BY_DATASET = {
        "german_credit": 'duration',
        "online_news_popularity": " shares",
        "covertype": "Horizontal_Distance_To_Hydrology",
        "online_shoppers": "ProductRelated",
        "bank_marketing": "pdays",
        "bank_loan": "Income",
        "diabetes": "target",
        "california_housing": "MedHouseVal",
        "king_county_housing": "price"

    }

    cont_datasets = {}
    cat_disc_datasets = {}
    mixed_datasets = {}

    print("Adding attributes to continuous data pkl...")
    dfs_cont_dict = pd.read_pickle("datasets/datasets_continuous_only.pkl")
    for ds_name, df in dfs_cont_dict.items():
        cont_datasets[ds_name] = prepare_smart_df(df, TARGET_COL_BY_DATASET[ds_name], "regression")

    print("Downloading Categorical Datasets")

    # 1. Student Perf (UCI 320)
    print("Fetching Student Performance...")
    student = fetch_ucirepo(id=320)
    df_student = pd.concat([student.data.features, student.data.targets], axis=1)
    cat_disc_datasets["Student Perf"] = prepare_smart_df(df_student, "G3", "classification")

    # 2. Lymphography (OpenML 10)
    print("Fetching Lymphography...")
    lympho = fetch_openml(data_id=10, as_frame=True, parser="auto")
    cat_disc_datasets["Lymphography"] = prepare_smart_df(lympho.frame, "class", "classification")

    # 3. Breast Cancer (UCI 14)
    print("Fetching Breast Cancer...")
    breast = fetch_ucirepo(id=14)
    df_breast = pd.concat([breast.data.features, breast.data.targets], axis=1)
    cat_disc_datasets["Breast cancer"] = prepare_smart_df(df_breast, "Class", "classification")

    # 4. Car Evaluation (UCI 19)
    print("Fetching Car Evaluation...")
    car = fetch_ucirepo(id=19)
    df_car = pd.concat([car.data.features, car.data.targets], axis=1)
    cat_disc_datasets["Car Evaluation"] = prepare_smart_df(df_car, "class", "classification")

    # 5. Mushroom (UCI 73)
    print("Fetching Mushroom...")
    mushroom = fetch_ucirepo(id=73)
    df_mush = pd.concat([mushroom.data.features, mushroom.data.targets], axis=1)
    cat_disc_datasets["Mushroom"] = prepare_smart_df(df_mush, "poisonous", "classification")

    print("\nDownloading Mixed Datasets")

    # 1. Adult (UCI 2)
    print("Fetching Adult...")
    adult = fetch_ucirepo(id=2)
    df_adult = pd.concat([adult.data.features, adult.data.targets], axis=1)
    mixed_datasets["Adult"] = prepare_smart_df(df_adult, "income", "classification")

    # 2. Credit Approval (UCI 27)
    print("Fetching Credit Approval...")
    credit = fetch_ucirepo(id=27)
    df_credit = pd.concat([credit.data.features, credit.data.targets], axis=1)
    mixed_datasets["Credit Approval"] = prepare_smart_df(df_credit, "A16", "classification")

    # 3. Online Shoppers (UCI 468)
    print("Fetching Online Shoppers...")
    shoppers = fetch_ucirepo(id=468)
    df_shoppers = pd.concat([shoppers.data.features, shoppers.data.targets], axis=1)
    mixed_datasets["Online Shoppers"] = prepare_smart_df(df_shoppers, "Revenue", "classification")

    # 4. Eucalyptus (OpenML 188)
    print("Fetching Eucalyptus...")
    eucalyptus = fetch_openml(data_id=188, as_frame=True, parser="auto")
    mixed_datasets["Eucalyptus"] = prepare_smart_df(eucalyptus.frame, "Utility", "classification")

    # 5. Forest Fires (UCI 162)
    print("Fetching Forest Fires...")
    fires = fetch_ucirepo(id=162)
    df_fires = pd.concat([fires.data.features, fires.data.targets], axis=1)
    mixed_datasets["Forest Fires"] = prepare_smart_df(df_fires, "area", "regression")

    # 6. Insurance (Kaggle)
    print("Fetching Insurance...")
    df_insurance = load_kaggle_csv("mirichoi0218/insurance")
    mixed_datasets["Insurance"] = prepare_smart_df(df_insurance, "charges", "regression")

    # 7. House Sales in King County (Kaggle)
    print("Fetching House Sales in King County...")
    df_houses = load_kaggle_csv("harlfoxem/housesalesprediction")
    df_houses = df_houses.drop(columns=['id'])
    mixed_datasets["House Sales"] = prepare_smart_df(df_houses, "price", "regression")

    # 8. Cardiovascular Disease (Kaggle)
    print("Fetching Cardiovascular Disease...")
    df_cardio = load_kaggle_csv("sulianova/cardiovascular-disease-dataset")
    df_cardio = df_cardio.drop(columns=['id'])
    mixed_datasets["Cardiovascular Disease"] = prepare_smart_df(df_cardio, "cardio", "classification")

    # 9. Churn Modelling (Kaggle)
    print("Fetching Churn Modelling...")
    df_churn = load_kaggle_csv("shrutimechlearn/churn-modelling")
    df_churn = df_churn.drop(columns=['RowNumber', 'CustomerId', 'Surname'])
    mixed_datasets["Churn Modelling"] = prepare_smart_df(df_churn, "Exited", "classification")

    # 10. Auto MPG (Regression)
    print("Fetching Auto MPG...")
    auto_mpg = fetch_ucirepo(id=9)
    df_auto_mpg = pd.concat([auto_mpg.data.features, auto_mpg.data.targets], axis=1)
    mixed_datasets["Auto MPG"] = prepare_smart_df(df_auto_mpg, "mpg", "regression")

    # 11. Diamonds (Kaggle)
    print("Fetching Diamonds...")
    df_diamonds = load_kaggle_csv("shivam2503/diamonds")
    df_diamonds = df_diamonds.drop(columns=['Unnamed: 0'], errors='ignore')
    mixed_datasets["Diamonds"] = prepare_smart_df(df_diamonds, "price", "regression")

    # 12. Real Estate (Kaggle)
    print("Fetching Real Estate Valuation...")
    df_real_estate = load_kaggle_csv("quantbruce/real-estate-price-prediction")
    df_real_estate = df_real_estate.drop(columns=['No'], errors='ignore')
    mixed_datasets["Real Estate"] = prepare_smart_df(df_real_estate, "Y house price of unit area", "regression")

    # 13. Stroke prediction (Kaggle)
    print("Fetching Stroke Prediction...")
    df_stroke = load_kaggle_csv("fedesoriano/stroke-prediction-dataset")
    df_stroke = df_stroke.drop(columns=['id'])
    mixed_datasets["Stroke Prediction"] = prepare_smart_df(df_stroke, "stroke", "classification")

    # 14. Palmer Penguins (Kaggle)
    print("Fetching Palmer Penguins...")
    df_penguins = load_kaggle_csv("parulpandey/palmer-archipelago-antarctica-penguin-data")
    df_penguins = df_penguins.drop(
        columns=['studyName', 'Sample Number', 'Individual ID', 'Region', 'Stage', 'Comments'], errors='ignore')
    mixed_datasets["Palmer Penguins"] = prepare_smart_df(df_penguins, "Species", "classification")

    print("\nSaving files...")
    save_pickle(cont_datasets, "datasets/datasets_continuous.pkl")
    save_pickle(cat_disc_datasets, 'datasets/datasets_categorical.pkl')
    save_pickle(mixed_datasets, 'datasets/datasets_mixed.pkl')

    print("Successfully created datasets/datasets_categorical.pkl and datasets/datasets_mixed.pkl.")

if __name__ == "__main__":
    main()