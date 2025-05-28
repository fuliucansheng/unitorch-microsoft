import os
import shutil
import hashlib
import time
import logging
import fire


def log(msg):
    print(msg)
    logging.info(msg)

def md5(file_path, chunk_size=4096):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def copy_and_verify(src_dir, dst_dir):
    for root, _, files in os.walk(src_dir):
        for file in files:
            src_file = os.path.join(root, file)
            rel_path = os.path.relpath(src_file, src_dir)
            dst_file = os.path.join(dst_dir, rel_path)

            os.makedirs(os.path.dirname(dst_file), exist_ok=True)
            if os.path.exists(dst_file):
                if md5(src_file) == md5(dst_file):
                    log(f"Verified OK: {rel_path} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
                    continue
                else:
                    log(f"MD5 mismatch, re-copy: {rel_path} at {time.strftime('%Y-%m-%d %H:%M:%S')}")

            shutil.copy2(src_file, dst_file)

            if md5(src_file) != md5(dst_file):
                log(f"ERROR: MD5 mismatch after copy: {rel_path} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
                if os.path.exists(dst_file):
                    os.remove(dst_file)
            else:
                log(f"Copied and verified: {rel_path} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            


def remove_old_ckpts(target_dir, keep_top_n):
    checkpoints = os.listdir(target_dir)
    checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
    if len(checkpoints) >= keep_top_n:
        num_to_remove = len(checkpoints) - keep_top_n + 1
        removing_checkpoints = checkpoints[0:num_to_remove]
        log(f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints")
        try:
            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(target_dir, removing_checkpoint)
                shutil.rmtree(removing_checkpoint,ignore_errors=True)
                log(f"Remove {removing_checkpoint}... at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        except:
            log(f"Failed to remove checkpoints: {removing_checkpoints}")



def main(SOURCE_DIR, TARGET_DIR, CHECK_INTERVAL=60, KEEP_TOP_N=3, LOG_FILE="ckpt_upload.log"):
    RANK = int(os.environ.get("RANK", 0))  # 获取当前进程的 RANK
    print(f"RANK: {RANK}")
    LOG_FILE = os.path.join(TARGET_DIR, LOG_FILE)

    os.makedirs(TARGET_DIR, exist_ok=True)

    logging.basicConfig(
        filename=LOG_FILE,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


    log("=== Start checkpoint watcher ===")
    seen_checkpoints = set()


    while True:
        try:
            if RANK == 0:
                remove_old_ckpts(TARGET_DIR, KEEP_TOP_N)
            for entry in sorted(os.listdir(SOURCE_DIR)):
                if not entry.startswith("checkpoint-"):
                    continue
                if entry in seen_checkpoints:
                    continue

                src_ckpt = os.path.join(SOURCE_DIR, entry)
                if not os.path.isdir(src_ckpt):
                    continue

                dst_ckpt = os.path.join(TARGET_DIR, entry)
                log(f"Processing {entry}... at {time.strftime('%Y-%m-%d %H:%M:%S')}")
                copy_and_verify(src_ckpt, dst_ckpt)

                seen_checkpoints.add(entry)

        except Exception as e:
            log(f"ERROR: {e}")

        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    fire.Fire()
