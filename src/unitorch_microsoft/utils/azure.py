# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import time
import hashlib
from typing import Optional
from azure.storage.blob import ContainerClient
from azure.storage.blob.aio import ContainerClient as AioContainerClient


class BlobFile:
    def __init__(
        self,
        path: str,
        container: ContainerClient,
    ):
        self.blob = container.get_blob_client(path)
        if self.blob.exists():
            self.info = self.blob.get_blob_properties()
        else:
            self.info = None

    def reset(self, blob_type: Optional[str] = "BLOCKBLOB"):
        assert blob_type in ["BLOCKBLOB", "APPENDBLOB"]

        if self.blob.info:
            self.blob.delete_blob()

        if blob_type == "APPENDBLOB":
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
