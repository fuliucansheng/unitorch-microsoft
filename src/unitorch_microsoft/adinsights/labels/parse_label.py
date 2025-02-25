import pandas as pd
import fire
import json
import time
import requests
import os 

from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.inference.models import SystemMessage, UserMessage


AZURE_ENDPOINT_URL = "https://DeepSeek-R1-fizvr.eastus2.models.ai.azure.com"
AZURE_ENDPOINT_KEY = os.getenv('DEEPSEEK_API_KEY')
llm_client = ChatCompletionsClient(
	endpoint=AZURE_ENDPOINT_URL,
	credential=AzureKeyCredential(AZURE_ENDPOINT_KEY)
)

def read_csv(file_path):
    """
    Reads a CSV file into a pandas DataFrame, filters out rows where the 'Label' column is empty,
    and adds a 'pair_tag' column by concatenating 'tag1' and 'tag2'.
    
    :param file_path: str, path to the CSV file
    :return: DataFrame
    """
    columns = ["AdId", "AdCopy", "FinalURL", "start_frame", "imgname", "neg_prompt", "prompt1", 
               "video1", "tag1", "prompt2", "video2", "tag2", "group", "name", "Index", "User", 
               "Comment", "Label"]
    df = pd.read_csv(file_path, names=columns, skiprows=1, sep='\t')  # Assuming the first row is the header
    df = df[df['Label'].notna() & (df['Label'] != '')]  # Filter out rows where 'Label' is NaN or an empty string
    df['pair_tag'] = df['tag1'].astype(str) + '+' +df['tag2'].astype(str)  # Concatenate 'tag1' and 'tag2'
    return df

def count_users_labeled(df):
    """
    Counts how many unique Users have labeled in the given DataFrame.
    
    :param df: DataFrame, the input DataFrame
    :return: int, number of unique Users who have labeled
    """
    return df['User'].nunique()

def calculate_label_distribution(df):
    """
    Calculates the distribution of labels in the given DataFrame.
    
    :param df: DataFrame, the input DataFrame
    :return: Series, distribution of labels
    """
    return df['Label'].value_counts()

def deepseek(query):
    System = """
        I will give you a set of annotation results. Each annotation result is numbered and includes a label and a comment. 
        This annotation compares which side, left or right, is better. It includes several cases: The left is better; 
        The right is better; Tie; All bad. Please help me analyze and summarize what issues are mentioned for the left 
        and right elements in this set of annotations. 
    """
    message = [
        {
            "role":"system",
            "content": System,
        },
        {
            "role":"user",
            "content":"Here are annotation results: {query}".format(query=query),
        },
    ]
    response = llm_client.complete(
        messages=message
        ).choices[0].message.content.replace("\t", " ").replace('\r', ' ')
    thinking, answer = response.split("</think>")
    thinking = thinking.strip("<think>").strip("\n").replace("\n", "#N#")
    answer = answer.strip("\n").replace("\n", " ")
    return answer



def analyze_comments(df):
    """
    Analyzes the given DataFrame piece to count rows with non-empty comments and put the comments in a list.
    
    :param df: DataFrame, the input DataFrame
    :return: tuple, (int, list) number of rows with non-empty comments and the list of comments
    """
    non_empty_comments = df[df['Comment'].notna() & (df['Comment'] != '')]
    comments_list = [f"{i}: (Label: {label})(Comment: {comment})" for i, (comment, label) in enumerate(zip(non_empty_comments['Comment'].tolist(), non_empty_comments['Label'].tolist()))]
    #print("comments: ", len(comments_list))
    comments = ';'.join(comments_list)
    answer = deepseek(comments)
    return answer


def analyze(
        file_path:str
        ):
    """
    Main function to read the CSV file, group by 'pair_tag', and count users and label distribution.
    
    :param file_path: str, path to the CSV file
    """
    print(file_path)
    df = read_csv(file_path)
    total_rows = len(df)
    unique_users = count_users_labeled(df)
    print(f"Total rows: {total_rows}")
    print(f"Number of unique users who labeled: {unique_users}")
    print("\n")
    print("\n")

    grouped = df.groupby('pair_tag')
    for name, group in grouped:
        row_count = len(group)
        user_count = count_users_labeled(group)
        label_distribution = calculate_label_distribution(group)
        comment = analyze_comments(group)
        print(f"Pair tag: {name}")
        print(f"Number of rows in group: {row_count}")
        print(f"Number of unique users who labeled: {user_count}")
        print("Label distribution:")
        print(label_distribution)
        print(f"comment: {comment}")
        print("\n")
        print("\n")

if __name__ == "__main__":
    fire.Fire()