import os
import zipfile
import urllib.request
import sys


DATASET_URL = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"
ZIP_FILE = "kagglecatsanddogs_5340.zip"
EXTRACT_DIR = "dataset"


def download_file(url, filename):
   
    print(f"Downloading from:\n  {url}")
    print(f"Saving to: {filename}")
    print("This may take a few minutes (~800 MB)...\n")

    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, downloaded * 100 / total_size)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            sys.stdout.write(
                f"\r  Progress: {percent:5.1f}%  ({mb_downloaded:.1f} / {mb_total:.1f} MB)"
            )
            sys.stdout.flush()

    urllib.request.urlretrieve(url, filename, reporthook)
    print("\n\nDownload complete!")


def extract_and_organize(zip_path, extract_dir):
    
    print(f"\nExtracting {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    print("Extraction complete!")

  
    src_cat = os.path.join(extract_dir, "PetImages", "Cat")
    src_dog = os.path.join(extract_dir, "PetImages", "Dog")
    dst_base = os.path.join(extract_dir, "training_set", "training_set")
    dst_cat = os.path.join(dst_base, "cats")
    dst_dog = os.path.join(dst_base, "dogs")

    os.makedirs(dst_cat, exist_ok=True)
    os.makedirs(dst_dog, exist_ok=True)

    print("\nOrganizing files into expected structure...")

    
    if os.path.isdir(src_cat):
        count = 0
        for fname in os.listdir(src_cat):
            src = os.path.join(src_cat, fname)
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                dst = os.path.join(dst_cat, f"cat.{count}.jpg")
                os.rename(src, dst)
                count += 1
        print(f"  Cats: {count} images")

    
    if os.path.isdir(src_dog):
        count = 0
        for fname in os.listdir(src_dog):
            src = os.path.join(src_dog, fname)
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                dst = os.path.join(dst_dog, f"dog.{count}.jpg")
                os.rename(src, dst)
                count += 1
        print(f"  Dogs: {count} images")

    print("\nDataset ready!")
    print(f"  Structure: {dst_base}/cats/ and {dst_base}/dogs/")


def main():
    if not os.path.exists(ZIP_FILE):
        download_file(DATASET_URL, ZIP_FILE)
    else:
        print(f"ZIP file already exists: {ZIP_FILE}")

    extract_and_organize(ZIP_FILE, EXTRACT_DIR)

    
    try:
        os.remove(ZIP_FILE)
        print(f"Cleaned up: {ZIP_FILE}")
    except OSError:
        pass

    print("\n Done! You can now run:  python svm_cats_dogs.py")


if __name__ == "__main__":
    main()
