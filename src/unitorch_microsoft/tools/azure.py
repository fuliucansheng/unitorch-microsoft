# Copyright (c) LIGHTINSIGHTER.
# Licensed under the MIT License.

import os
import fire
import logging
import time
import hashlib
from typing import Optional
from azure.storage.blob import ContainerClient
from azure.storage.blob.aio import ContainerClient as AioContainerClient

logging.getLogger("azure").propagate = False
logging.getLogger("azure.storage").propagate = False


class Container:
    def __init__(
        self,
        name: str,
        url: str = os.environ.get("AZURE_URL"),
        key: str = os.environ.get("AZURE_KEY"),
    ):
        self.container = ContainerClient(
            url,
            name,
            key,
        )

    def list(self, folder: Optional[str] = None):
        return self.container.list_blobs(name_starts_with=folder)

    def delete(self, folder):
        for blob in self.container.list_blobs(name_starts_with=folder):
            self.container.delete_blob(blob.name)


class BlobFile:
    def __init__(
        self,
        path: str,
        name: str,
        url: str = os.environ.get("AZURE_URL"),
        key: str = os.environ.get("AZURE_KEY"),
    ):
        container = ContainerClient(
            url,
            name,
            key,
        )
        self.blob = container.get_blob_client(path)
        if self.blob.exists():
            self.info = self.blob.get_blob_properties()
        else:
            self.info = None

    def reset(self, blob_type: Optional[str] = "BLOCKBLOB"):
        assert blob_type in ["BLOCKBLOB", "AppendBlob"]

        if self.blob.exists():
            self.blob.delete_blob()

        if blob_type == "AppendBlob":
            self.blob.create_append_blob()

        return self

    @staticmethod
    def hexdigest(data_bytes: Optional[bytes] = None):
        if data_bytes is None:
            timestamp = str(time.time()).encode()
            hexdigest = hashlib.sha256(timestamp).hexdigest()
        else:
            hexdigest = hashlib.sha256(data_bytes).hexdigest()

        return hexdigest

    def upload(
        self,
        data_bytes: Optional[bytes] = None,
        path: Optional[str] = None,
    ):
        assert data_bytes is not None or path is not None
        if path:
            data_bytes = open(path, "rb").read()
        self.blob.upload_blob(data_bytes, overwrite=True)
        return self

    def append(self, data_bytes: bytes):
        self.blob.append_block(data_bytes)

    def delete(self):
        self.blob.delete_blob()
        return self

    @property
    def size(self):
        return self.info.size

    @property
    def url(self):
        return self.blob.url

    def exists(self):
        return self.blob.exists()


class AzureTools:
    def __init__(
        self,
        url: str = os.environ.get("AZURE_URL"),
        key: str = os.environ.get("AZURE_KEY"),
    ):
        self.url = url
        self.key = key

    def delete_file(
        self,
        container: str,
        file: str,
    ):
        client = BlobFile(file, name=container, url=self.url, key=self.key)
        client.delete()
        logging.info(f"file deleted from {container}/{file} successfully.")

    def delete_folder(
        self,
        container: str,
        folder: str,
    ):
        client = Container(name=container, url=self.url, key=self.key)
        client.delete(folder)
        logging.info(f"file deleted from {container}/{folder} successfully.")

    def upload_file(
        self,
        container: str,
        local_file: str,
        remote_file: str,
    ):
        if not os.path.exists(local_file):
            raise ValueError(f"local file {local_file} not found.")

        client = BlobFile(remote_file, name=container, url=self.url, key=self.key)
        client.reset().upload(path=local_file)
        logging.info(
            f"file uploaded from {local_file} to {container}/{remote_file} successfully."
        )

    def upload_folder(
        self,
        container: str,
        local_folder: str,
        remote_folder: str,
    ):
        if not os.path.exists(local_folder):
            raise ValueError(f"local folder {local_folder} not found.")

        for root, _, files in os.walk(local_folder):
            for file in files:
                local_file = os.path.join(root, file)
                remote_file = os.path.join(
                    remote_folder, os.path.relpath(local_file, local_folder)
                )
                client = BlobFile(
                    remote_file, name=container, url=self.url, key=self.key
                )
                client.reset().upload(path=local_file)
                logging.info(
                    f"file uploaded from {local_file} to {container}/{remote_file} successfully."
                )


if __name__ == "__main__":
    fire.Fire(AzureTools)
