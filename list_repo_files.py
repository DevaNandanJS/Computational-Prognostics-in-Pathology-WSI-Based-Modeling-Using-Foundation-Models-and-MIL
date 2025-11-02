from huggingface_hub import HfFileSystem

try:
    print("Connecting to Hugging Face Hub to list files...")
    fs = HfFileSystem()
    repo_id = "datasets/MahmoodLab/UNI2-h-features"

    # The fs.ls() command lists the contents of a directory
    file_list = fs.ls(repo_id, detail=False)

    print("\n--- Repository Contents ---")
    for path in file_list:
        # We only want the part of the path after the repo_id
        print(path.replace(repo_id + '/', ''))
    print("---------------------------\n")

except Exception as e:
    print(f"An error occurred: {e}")