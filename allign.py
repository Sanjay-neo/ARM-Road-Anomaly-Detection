import os
import shutil

SRC = r"C:\Users\sanja\Downloads\hazard_in_road"
DST = r"D:\ARM_hackathon\road_anomaly"

SPLITS = ["train", "valid", "test"]
SUBFOLDERS = ["images", "labels"]


def ensure_dirs():
    for split in SPLITS:
        for sub in SUBFOLDERS:
            os.makedirs(os.path.join(DST, split, sub), exist_ok=True)


def copy_dataset():
    for split in SPLITS:
        for sub in SUBFOLDERS:
            src_dir = os.path.join(SRC, split, sub)
            dst_dir = os.path.join(DST, split, sub)

            if not os.path.isdir(src_dir):
                print(f"⚠️ Skipping missing folder: {src_dir}")
                continue

            for file in os.listdir(src_dir):
                src_file = os.path.join(src_dir, file)
                dst_file = os.path.join(dst_dir, file)

                if os.path.isfile(src_file):
                    shutil.copy(src_file, dst_file)

            print(f"✅ Copied {split}/{sub}")


if __name__ == "__main__":
    ensure_dirs()
    copy_dataset()
    print("\n🎯 hazard_in_road dataset copied successfully into road_anomaly.")
