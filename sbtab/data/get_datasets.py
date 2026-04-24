import pickle
import pandas as pd

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

def main():
    cat_disc_datasets = {}
    mixed_datasets = {}

    print("\nDownloading Only Categorical Datasets...")

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

    print("\nDownloading mixed datasets...")

    # 1. Adult (UCI 2)
    print("Fetching Adult...")
    adult = fetch_ucirepo(id=2)
    df_adult = pd.concat([adult.data.features, adult.data.targets], axis=1)
    mixed_datasets["Adult"] = prepare_smart_df(df_adult, "income", "regression")

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

    print("\nSaving files...")
    save_pickle(cat_disc_datasets, 'datasets/datasets_categorical.pkl')
    save_pickle(mixed_datasets, 'datasets/datasets_mixed.pkl')

    print("Successfully created datasets/datasets_categorical.pkl and datasets/datasets_mixed.pkl.")

if __name__ == "__main__":
    main()