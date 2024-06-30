import os
import json

import pandas as pd

DATA_PATH = os.getenv("DATA_PATH", default="data/titanic-training-data.csv")

class Preprocessor:
    DATA_PATH = os.getenv("DATA_PATH", default="data/titanic-training-data.csv")
    COLUMN_CONFIG = os.getenv("COLUMN_CONFIG", default="config/columns.json")
    TRAINING_COLUMNS_CONFIG = os.getenv("TRAINING_COLUMNS_CONFIG", default="config/training_columns.json")
    EMBARKED_CONFIG = os.getenv("EMBARKED_CONFIG", default="config/embarked.json")
    TITLE_CONFIG = os.getenv("TITLE_CONFIG", default="config/title.json")
    AGE_GROUP_CONFIG = os.getenv("AGE_GROUP_CONFIG", default="config/age_group.json")

    def __init__(self):
        self.data = pd.read_csv(DATA_PATH)
        self.data.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], inplace=True)
        self.columns = self.load_columns()
        self.training_columns = self.load_training_columns()
        self.embarked = self.load_embarked()
        self.title = self.load_title()
        self.age_group = self.load_age_group()
    
    @classmethod
    def load_columns(cls):
        with open(cls.COLUMN_CONFIG, "r") as f:
            return json.load(f)
    
    @classmethod
    def load_training_columns(cls):
        with open(cls.TRAINING_COLUMNS_CONFIG, "r") as f:
            return json.load(f)
    
    @classmethod
    def load_embarked(cls):
        with open(cls.EMBARKED_CONFIG, "r") as f:
            return json.load(f)
        
    @classmethod
    def load_title(cls):
        with open(cls.TITLE_CONFIG, "r") as f:
            return json.load(f)
        
    @classmethod
    def load_age_group(cls):
        with open(cls.AGE_GROUP_CONFIG, "r") as f:
            return json.load(f)
    
    @staticmethod
    def column_encoder(df: pd.DataFrame, config: dict, column_name: str) -> pd.DataFrame:
        """
        Creates new columns based on the key found in the specified column of the DataFrame,
        sets one column to True based on the value in the specified column, and the others to False.
        The original column is then dropped from the DataFrame.

        Parameters:
            df (pd.DataFrame): The DataFrame to modify.
            config (dict): A configuration dictionary mapping keys (values of the column) to new DataFrame column names.
            column_name (str): The name of the column in the DataFrame to use for setting the new columns.

        Returns:
            pd.DataFrame: The modified DataFrame with the new columns added and the original column removed.
        """
        # Ensure that the column exists in the DataFrame
        if column_name not in df.columns:
            raise ValueError(f"The column '{column_name}' does not exist in the DataFrame.")
        
        # Ensure that the configuration dictionary is not empty
        if not config:
            raise ValueError("The configuration dictionary cannot be empty.")
        
        # Ensure that the value of the column is a key in the configuration dictionary
        for index, row in df.iterrows():
            if row[column_name] not in config:
                raise ValueError(f"The value '{row[column_name]}' in the column '{column_name}' is not a valid key in the configuration dictionary.")
        
        # Initialize all specified columns to False
        for key in config.keys():
            df[config[key]] = False
        
        # Set the appropriate new column to True based on the original column's value
        for index, row in df.iterrows():
            if row[column_name] in config:
                df.loc[index, config[row[column_name]]] = True

        # Drop the original column
        df.drop(column_name, axis=1, inplace=True)
        
        return df
    
    @staticmethod
    def reorder_and_cast_columns(df: pd.DataFrame, reference_dict: dict) -> pd.DataFrame:
        """
        Reorder the columns of df to match the keys of reference_dict and cast them to the specified types.

        Parameters:
            df (pd.DataFrame): The DataFrame whose columns need to be reordered and types cast.
            reference_dict (dict): A dictionary where keys are column names and values are the desired data types.

        Returns:
            pd.DataFrame: A DataFrame with columns reordered and cast to match the reference dictionary.
        """
        # Extract the column order from the dictionary keys
        column_order = list(reference_dict.keys())

        # Reorder the columns based on the dictionary keys
        reordered_df = df.reindex(columns=column_order)

        # Cast the columns to the specified types in reference_dict
        for column, dtype in reference_dict.items():
            if column in reordered_df.columns:  # check if the column is in the DataFrame
                reordered_df[column] = reordered_df[column].astype(dtype)

        return reordered_df

    def preprocess(self, input_df: pd.DataFrame=None) -> pd.DataFrame:
        """
        Preprocess the data
        """
        df = input_df.copy()
        # print(df.head())
        # Create a new column indicating whether the passenger has a cabin
        df['Cabin_Ind'] = df['Cabin'].notnull().astype(int)

        # Encode "Sex" feature
        df['Sex'] = df["Sex"].map(
            {
                "male": 1,
                "female": 0,
            }
        )

        # One-Hot Encoding of 'Embarked'
        embarked_dummies = pd.get_dummies(df['Embarked'], prefix='Embarked')
        df = pd.concat([df, embarked_dummies], axis=1)

        # Feature 1: Extract titles from names

        # Feature 2: Create family size feature
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

        # Feature 3: Create a feature for solo travellers
        df['IsAlone'] = 0
        df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1

        # Feature 4: Categorize age into groups
        df = self.column_encoder(df, self.title, "Title")
        df = self.column_encoder(df, self.embarked, "Embarked")
        
        bins = [0, 12, 20, 40, 60, 80, float('inf')]
        labels = ['Child', 'Teen', 'Adult', 'Middle-aged', 'Senior', "Over-aged"]
        df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
        df = self.column_encoder(df, self.age_group, "AgeGroup")

        # Feature 5: Calculate fare per person
        df['FarePerPerson'] = df['Fare'] / df['FamilySize']

        # Dropping non-used columns
        columns_to_drop = ['Ticket', 'Cabin', 'Fare']
        df = df.drop(columns=columns_to_drop)

        return self.reorder_and_cast_columns(df=df, reference_dict=self.training_columns)
