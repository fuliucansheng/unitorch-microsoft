import pandas as pd
import fire
import json
import time
import requests
import os

from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.inference.models import SystemMessage, UserMessage
import random


AZURE_ENDPOINT_URL = "https://DeepSeek-R1-fizvr.eastus2.models.ai.azure.com"
AZURE_ENDPOINT_KEY = os.getenv("DEEPSEEK_API_KEY")
llm_client = ChatCompletionsClient(
    endpoint=AZURE_ENDPOINT_URL, credential=AzureKeyCredential(AZURE_ENDPOINT_KEY)
)


def read_csv(file_path):
    """
    Reads a CSV file into a pandas DataFrame, filters out rows where the 'Label' column is empty,
    and adds a 'pair_tag' column by concatenating 'tag1' and 'tag2'.

    :param file_path: str, path to the CSV file
    :return: DataFrame
    """
    columns = [
        "AdId",
        "AdCopy",
        "FinalURL",
        "start_frame",
        "imgname",
        "neg_prompt",
        "prompt1",
        "video1",
        "tag1",
        "prompt2",
        "video2",
        "tag2",
        "group",
        "name",
        "Index",
        "User",
        "Comment",
        "Label",
    ]
    df = pd.read_csv(
        file_path, names=columns, skiprows=1, sep="\t"
    )  # Assuming the first row is the header
    df = df[
        df["Label"].notna() & (df["Label"] != "")
    ]  # Filter out rows where 'Label' is NaN or an empty string
    df["pair_tag"] = (
        df["tag1"].astype(str) + "+" + df["tag2"].astype(str)
    )  # Concatenate 'tag1' and 'tag2'
    return df


def filter_frames(df, tag1_value, tag2_value):
    """
    Filters the DataFrame based on the given conditions for 'tag1' and 'tag2'.

    :param df: DataFrame, the input DataFrame
    :param tag1_value: str, the value to filter 'tag1' column
    :param tag2_value: str, the value to filter 'tag2' column
    :return: DataFrame, filtered DataFrame
    """
    filtered_df = df[(df["tag1"] == tag1_value) & (df["tag2"] == tag2_value)]
    filtered_df = filtered_df.drop(columns=["pair_tag"])
    filtered_df.to_csv("filtered_frames_results.tsv", sep="\t", index=False)
    filtered_df.drop(columns=["Index", "User", "Comment", "Label"]).to_csv(
        "filtered_frames.tsv", sep="\t", index=False
    )
    return filtered_df


def is_conflict(label1, label2):
    """
    Determines if there is a conflict between two labels.

    :param label1: str, the first label
    :param label2: str, the second label
    :return: bool, True if there is a conflict, False otherwise
    """
    conflict_pairs = [("Left is Better", "Right is Better"), ("Tie", "All Bad")]

    return (label1, label2) in conflict_pairs or (label2, label1) in conflict_pairs


def check_label_conflicts(df):
    """
    Groups rows with the same 'start_frame', 'tag1', and 'tag2', keeps groups with >= 2 rows,
    randomly samples 2 rows within each group, and checks if the labeling results conflict.
    Counts the overall conflict rate across all groups.

    :param df: DataFrame, the input DataFrame
    :return: float, overall conflict rate
    """
    grouped = df.groupby(["start_frame", "tag1", "tag2"])
    conflict_count = 0
    total_groups = 0

    for _, group in grouped:
        if len(group) >= 2:
            sampled_rows = group.sample(n=2, random_state=1)
            labels = sampled_rows["Label"].tolist()
            # if is_conflict(labels[0], labels[1]):
            if labels[0] == labels[1]:
                conflict_count += 1
            total_groups += 1

    if total_groups == 0:
        return 0.0

    conflict_rate = float(conflict_count) / float(total_groups)
    print(f"conflict_rate: {conflict_rate}")
    return conflict_rate


def count_users_labeled(df):
    """
    Counts how many unique Users have labeled in the given DataFrame.

    :param df: DataFrame, the input DataFrame
    :return: int, number of unique Users who have labeled
    """
    return df["User"].nunique()


def calculate_label_distribution(df):
    """
    Calculates the distribution of labels in the given DataFrame.

    :param df: DataFrame, the input DataFrame
    :return: Series, distribution of labels
    """
    return df["Label"].value_counts()


def deepseek(query):
    System = """
        I will give you a set of annotation results. This annotation involves generating a video from the same image using two different methods. The annotation is to determine which method is better. 
        It includes several cases: Left is Better;  Right is Better; Tie; All Bad.
        Each annotation result is numbered and includes a image (which image this annotation is about), a label (labeling result). 
        Here are two tasks need your help: 
        1) Please help me analyze and summarize what most frequently issues and benefits are mentioned for the left and right elements, respectively. Only need to dump top 1-3 items for each side. Warp result within <Issue></Issue>.
        2) Sometimes the annotation label is bad because the image is not suitable. Also, help me summarize what proportion of images are unsuitable. Warp result within <Image></Image>. 
    """
    #
    # 1) Clean the comment data and format the data. For each annotation, wrap comment related with left within <left> and </left>; wrap comment related with right within <right> and </right>; if it is about both, wrap it within <both> and </both>.
    message = [
        {
            "role": "system",
            "content": System,
        },
        {
            "role": "user",
            "content": "Here are annotation results: {query}".format(query=query),
        },
    ]
    response = (
        llm_client.complete(messages=message)
        .choices[0]
        .message.content.replace("\t", " ")
        .replace("\r", " ")
    )
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
    non_empty_comments = df[df["Comment"].notna() & (df["Comment"] != "")]
    comments_list = [
        f"{i}: (Image: {image})(Label: {label}; {comment})"
        for i, (comment, label, image) in enumerate(
            zip(
                non_empty_comments["Comment"].tolist(),
                non_empty_comments["Label"].tolist(),
                non_empty_comments["start_frame"].tolist(),
            )
        )
    ]
    # print("comments: ", len(comments_list))
    comments = ";".join(comments_list)
    print(comments)

    try:
        answer = ""  # deepseek(comments)
        print(answer)
    except:
        pass
    return None


def analyze(file_path: str):
    """
    Main function to read the CSV file, group by 'pair_tag', and count users and label distribution.

    :param file_path: str, path to the CSV file
    """
    print(file_path)
    df = read_csv(file_path)
    total_rows = len(df)
    unique_users = count_users_labeled(df)
    check_label_conflicts(df)
    print(f"Total rows: {total_rows}")
    print(f"Number of unique users who labeled: {unique_users}")
    print("\n")
    print("\n")

    grouped = df.groupby("pair_tag")
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
    filter_frames(df, "Keling-NoPrompt", "Keling-Gpt4o")


if __name__ == "__main__":
    fire.Fire()
