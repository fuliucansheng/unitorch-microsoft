import fire
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import re
import pandas as pd
import multiprocessing as mp
import os
import numpy as np
import io


def process_chunk(
    videos,
    chunk_start,
    chunk_size,
    process_id,
    cache_dir,
    num_processes,
    total_rows,
    file_writer,
    lock,
):
    if process_id == num_processes - 1:
        chunk_size = total_rows - chunk_start + 1
    chunks = videos[chunk_start : chunk_start + chunk_size]
    print(
        f"Worker {process_id} Processing rows: {chunk_start} to {chunk_start + chunk_size - 1} \n"
    )

    movement_str = ""
    for idx, video in enumerate(chunks):
        try:
            src_file = os.path.join("/datablob/shutterstock", video)
            if os.path.exists(src_file):
                dst_file = video.replace("/", "_")
                dst_file = os.path.join(cache_dir, dst_file)
                cmd = f"cp {src_file} {dst_file}"
                os.system(cmd)
                print(f"Worker {process_id} copy {src_file} to {dst_file}")
                movement_str += f"{video}\t{dst_file}\n"
                if idx % 100 == 0:
                    with lock:
                        writer = open(file_writer, "a+")
                        writer.write(movement_str)
                        writer.flush()
                        writer.close()
                    movement_str = ""
            else:
                print(f"Worker {process_id} file {src_file} not found")
                continue
        except Exception as e:
            print(f"Worker {process_id} error processing {video}: {e}")
            continue

    if movement_str != "":
        with lock:
            writer = open(file_writer, "a+")
            writer.write(movement_str)
            writer.flush()
            writer.close()


def azure_login(connect_key, account_name, container_name):
    """
    intall required packages: pip3 install azure-storage-blob azure-identity
    """
    from azure.storage.blob import BlobServiceClient
    from azure.storage.blob import BlobClient, ContentSettings

    connect_str = (
        "DefaultEndpointsProtocol=https;AccountName="
        + account_name
        + ";AccountKey="
        + connect_key
        + ";EndpointSuffix=core.windows.net"
    )
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    container_client = blob_service_client.get_container_client(container_name)
    return container_client


def get_azureurl(
    data, container_client, savename, container_name, subfolder, account_name
):
    from azure.storage.blob import BlobServiceClient
    from azure.storage.blob import BlobClient, ContentSettings
    import mimetypes

    try:
        remote_name = subfolder + "/" + savename
        content_type, _ = mimetypes.guess_type(remote_name)
        if content_type is None:
            content_type = "application/octet-stream"

        data_blob = container_client.get_blob_client(remote_name)
        with open(data, "rb") as data_file:
            data_blob.upload_blob(
                data_file,
                overwrite=True,
                content_settings=ContentSettings(content_type=content_type),
            )

        url = f"https://{account_name}.blob.core.windows.net/{container_name}/{remote_name}"

        return url
    except Exception as e:
        # print("upload img2azure failed {!r}".format(e))
        return None


def process_chunk_azure(
    videos,
    chunk_start,
    chunk_size,
    process_id,
    subfolder,
    num_processes,
    total_rows,
    file_writer,
    lock,
    account_name,
    container_name,
    container_client,
):
    if process_id == num_processes - 1:
        chunk_size = total_rows - chunk_start + 1
    chunks = videos[chunk_start : chunk_start + chunk_size]
    print(
        f"Worker {process_id} Processing rows: {chunk_start} to {chunk_start + chunk_size - 1} \n"
    )

    movement_str = ""
    for idx, video in enumerate(chunks):
        try:
            src_file = os.path.join("/datablob/shutterstock", video)
            if os.path.exists(src_file):
                dst_file = video.replace("/", "_")
                video_url = get_azureurl(
                    src_file,
                    container_client,
                    dst_file,
                    container_name,
                    subfolder,
                    account_name,
                )
                if video_url is None:
                    print(f"Worker {process_id} upload {src_file} to Azure failed")
                    continue
                movement_str += f"{video}\t{video_url}\n"
                if idx % 100 == 0:
                    with lock:
                        writer = open(file_writer, "a+")
                        writer.write(movement_str)
                        writer.flush()
                        writer.close()
                    movement_str = ""
            else:
                print(f"Worker {process_id} file {src_file} not found")
                continue
        except Exception as e:
            print(f"Worker {process_id} error processing {video}: {e}")
            continue

    if movement_str != "":
        with lock:
            writer = open(file_writer, "a+")
            writer.write(movement_str)
            writer.flush()
            writer.close()


def movement(
    data_file: str,
    dst_dir: str,
    names: Union[str, List[str]] = "video",
    move_col: str = "video",
    max_cnt: int = 100000000,
    sub_folder: str = "video",
    upload_to_azure: bool = False,
    connect_key: Optional[str] = None,
    account_name: Optional[str] = None,
    container_name: Optional[str] = None,
):
    """
    Movement of video data, used for video data movement.
    """

    if isinstance(names, str) and names.strip() == "*":
        names = None
    if isinstance(names, str):
        names = re.split(r"[,;]", names)
        names = [n.strip() for n in names]
    data = pd.read_csv(
        data_file,
        names=names,
        sep="\t",
        quoting=3,
        header=None,
        nrows=max_cnt,
    )
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir, exist_ok=True)
    output_file = f"{dst_dir}/output.tsv"
    if os.path.exists(output_file):
        uniques = []
        with open(output_file, "r") as f:
            for line in f.readlines():
                videoid, dst_name = line.strip().split("\t")
                uniques.append(videoid)
        data = data[
            ~data.apply(
                lambda x: x[move_col] in uniques,
                axis=1,
            )
        ]
    print(f"Data loaded, total rows to move: {len(data)}")
    videos = data[move_col].tolist()
    num_processes = mp.cpu_count()
    total_rows = len(videos)
    chunk_size = total_rows // num_processes

    dst_dir = os.path.join(dst_dir, sub_folder)

    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir, exist_ok=True)

    lock = mp.Lock()

    processes = []
    print(f"need to process {total_rows} videos")
    for i in range(num_processes):
        if upload_to_azure:
            if connect_key is None or account_name is None or container_name is None:
                raise ValueError(
                    "connect_key, account_name, and container_name must be provided for Azure upload."
                )

            container_client = azure_login(connect_key, account_name, container_name)
            p = mp.Process(
                target=process_chunk_azure,
                args=(
                    videos,
                    i * chunk_size,
                    chunk_size,
                    i,
                    sub_folder,
                    num_processes,
                    total_rows,
                    output_file,
                    lock,
                    account_name,
                    container_name,
                    container_client,
                ),
            )
        else:
            p = mp.Process(
                target=process_chunk,
                args=(
                    videos,
                    i * chunk_size,
                    chunk_size,
                    i,
                    dst_dir,
                    num_processes,
                    total_rows,
                    output_file,
                    lock,
                ),
            )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
    return


def process_check(
    videos,
    chunk_start,
    chunk_size,
    process_id,
    data_dir,
    num_processes,
    total_rows,
    file_writer,
    lock,
):
    if process_id == num_processes - 1:
        chunk_size = total_rows - chunk_start + 1
    chunks = videos[chunk_start : chunk_start + chunk_size]
    print(
        f"Worker {process_id} Processing rows: {chunk_start} to {chunk_start + chunk_size - 1} \n"
    )

    res_str = ""
    exist_cnt = 0
    for idx, video in enumerate(chunks):
        try:
            if os.path.exists(os.path.join(data_dir, video)):
                res_str += f"{video}\n"
                exist_cnt += 1
                if idx % 100 == 0:
                    with lock:
                        writer = open(file_writer, "a+")
                        writer.write(res_str)
                        writer.flush()
                        writer.close()
                    res_str = ""
            else:
                continue
        except Exception as e:
            continue

    if res_str != "":
        with lock:
            writer = open(file_writer, "a+")
            writer.write(res_str)
            writer.flush()
            writer.close()
    print(
        f"Worker {process_id} processed {len(chunks)} videos, found {exist_cnt} existing videos."
    )


def checkexists(
    data_file: str,
    data_dir: str,
    dst_dir: str,
    names: Union[str, List[str]] = "video",
    move_col: str = "video",
    max_cnt: int = 100000000,
):
    """
    Movement of video data, used for video data movement.
    """

    if isinstance(names, str) and names.strip() == "*":
        names = None
    if isinstance(names, str):
        names = re.split(r"[,;]", names)
        names = [n.strip() for n in names]
    data = pd.read_csv(
        data_file,
        names=names,
        sep="\t",
        quoting=3,
        header=None,
        nrows=max_cnt,
    )
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir, exist_ok=True)
    output_file = f"{dst_dir}/output.tsv"
    if os.path.exists(output_file):
        uniques = []
        with open(output_file, "r") as f:
            for line in f.readlines():
                videoid = line.strip().split("\t")
                uniques.append(videoid)
        data = data[
            ~data.apply(
                lambda x: x[move_col] in uniques,
                axis=1,
            )
        ]
    print(f"Data loaded, total rows to check: {len(data)}")
    videos = data[move_col].tolist()
    num_processes = mp.cpu_count()
    total_rows = len(videos)
    chunk_size = total_rows // num_processes

    lock = mp.Lock()

    processes = []
    print(f"need to process {total_rows} videos")
    for i in range(num_processes):
        p = mp.Process(
            target=process_check,
            args=(
                videos,
                i * chunk_size,
                chunk_size,
                i,
                data_dir,
                num_processes,
                total_rows,
                output_file,
                lock,
            ),
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
    return


if __name__ == "__main__":
    fire.Fire()
