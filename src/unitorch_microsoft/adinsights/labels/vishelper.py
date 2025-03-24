import pandas as pd
import fire


def read_and_filter_tsv(file_path, columns_to_keep, column_names):
    """
    Reads a TSV file, renames the columns, and keeps only the specified columns.

    Parameters:
    file_path (str): The path to the TSV file.
    columns_to_keep (list): List of columns to keep.
    new_column_names (dict): Dictionary mapping old column names to new column names.

    Returns:
    pd.DataFrame: The filtered DataFrame with renamed columns.
    """
    # Read the TSV file
    df = pd.read_csv(file_path, names=column_names, skiprows=1, sep="\t")

    # Keep only the specified columns
    df = df[columns_to_keep]

    return df


def read_and_join_tsv_files(
    file_paths, columns_to_keep, column_names, join_key, join_column_names
):
    """
    Reads a list of TSV files, filters and renames columns, and performs a left outer join on the specified key.

    Parameters:
    file_paths (str): String of paths to the TSV files, concatenated with common separators like ",;".
    columns_to_keep (str): String of columns to keep, concatenated with ",;".
    column_names (str): String of old column names to new column names, concatenated with ",;".
    join_key (str): The key column to join on.
    join_column_names (str): String of columns to keep from each join, concatenated with ",;".

    Returns:
    pd.DataFrame: The joined DataFrame.
    """
    # Split the file_paths string into a list of file paths
    file_paths_list = [path.strip() for path in file_paths.split(";")]

    # Split the columns_to_keep string into a list
    columns_to_keep_list = [col.strip() for col in columns_to_keep.split(";")]

    # Split the column_names string into a dictionary
    column_names_list = [item.strip() for item in column_names.split(";")]

    # Split the join_column_names string into a list
    join_column_names_list = [col.strip() for col in join_column_names.split(";")]

    # Initialize the DataFrame with the first file
    df = read_and_filter_tsv(
        file_paths_list[0], columns_to_keep_list, column_names_list
    )
    print("Column names of the initial DataFrame:", df.columns)
    df2 = read_and_filter_tsv(
        file_paths_list[1], columns_to_keep_list, column_names_list
    )
    print("Column names of the 2nd DataFrame:", df2.columns)
    df_merge = df.merge(
        df2[join_column_names_list], on=join_key, how="left", suffixes=("_0", "_1")
    )
    print("Column names after merge:", df_merge.columns)

    # Iterate over the remaining files and perform left outer join
    for cnt in range(2, len(file_paths_list)):
        file_path = file_paths_list[cnt]
        temp_df = read_and_filter_tsv(
            file_path, columns_to_keep_list, column_names_list
        )
        df_merge = df_merge.merge(
            temp_df[join_column_names_list],
            on=join_key,
            how="left",
            suffixes=("", f"_{cnt}"),
        )
        print("Column names after merge:", df_merge.columns)

    # Dump the DataFrame to a TSV file with header
    df_merge.to_csv("output.tsv", sep="\t", index=False, header=True)

    return df


if __name__ == "__main__":
    fire.Fire()
