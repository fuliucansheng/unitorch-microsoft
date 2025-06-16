import fire
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import re
import pandas as pd
import multiprocessing as mp
import os


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


def movement(
    data_file: str,
    dst_dir: str,
    names: Union[str, List[str]] = "video",
    move_col: str = "video",
    max_cnt: int = 100000000,
    sub_folder: str = "video",
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
    print(f"Worker {process_id} processed {len(chunks)} videos, found {exist_cnt} existing videos.")

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
