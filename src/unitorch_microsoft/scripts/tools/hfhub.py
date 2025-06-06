# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import fire
import json
import subprocess
import logging
from huggingface_hub import (
    login,
    delete_file,
    delete_folder,
    upload_file,
    upload_folder,
    list_repo_files,
)


class HFHubTools:
    def __init__(
        self,
        repo,
        token=None,
        repo_type="model",
        new_session=False,
    ):
        login(token=token, new_session=new_session)
        self.repo = repo
        self.repo_type = repo_type

    def delete_file(
        self,
        file: str,
    ):
        delete_file(
            path_in_repo=file,
            repo_id=self.repo,
            repo_type=self.repo_type,
            commit_message=f"Delete file {file}",
        )
        logging.info(f"file deleted from {self.repo}/{file} successfully.")

    def delete_folder(
        self,
        folder: str,
    ):
        delete_folder(
            path_in_repo=folder,
            repo_id=self.repo,
            repo_type=self.repo_type,
            commit_message=f"Delete file {folder}",
        )
        logging.info(f"folder deleted from {self.repo}/{folder} successfully.")

    def upload_file(
        self,
        local_file: str,
        remote_file: str,
    ):
        if not os.path.exists(local_file):
            raise ValueError(f"local file {local_file} not found.")

        upload_file(
            path_or_fileobj=local_file,
            path_in_repo=remote_file,
            repo_id=self.repo,
            repo_type=self.repo_type,
            commit_message=f"Upload file {remote_file}",
        )

        logging.info(
            f"file uploaded from {local_file} to {self.repo}/{remote_file} successfully."
        )

    def upload_folder(
        self,
        local_folder: str,
        remote_folder: str,
    ):
        if not os.path.exists(local_folder):
            raise ValueError(f"local folder {local_folder} not found.")

        upload_folder(
            folder_path=local_folder,
            repo_id=self.repo,
            repo_type=self.repo_type,
            path_in_repo=remote_folder,
            commit_message=f"Upload folder {remote_folder}",
        )
        logging.info(
            f"folder uploaded from {local_folder} to {self.repo}/{remote_folder} successfully."
        )


if __name__ == "__main__":
    fire.Fire(HFHubTools)
