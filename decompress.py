import os
import gzip
import shutil

def decompress_gz_files(directory):
    # Walk through the directory and subdirectories
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.gz'):
                gz_file_path = os.path.join(root, file)
                decompressed_file_path = os.path.join(root, file[:-3])  # Remove the .gz extension

                # Decompress the file
                with gzip.open(gz_file_path, 'rb') as f_in:
                    with open(decompressed_file_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                
                print(f"Decompressed: {gz_file_path} to {decompressed_file_path}")

                # Optionally, delete the .gz file after decompression
                os.remove(gz_file_path)

if __name__ == "__main__":
    directory_path = '.'  # Replace with your directory path
    decompress_gz_files(directory_path)
