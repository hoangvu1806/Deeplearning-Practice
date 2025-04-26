import argparse
import os
import tqdm

from utils import read_file, create_dir 
from utils import write_content


def crawl_urls(urls_fpath="urls.txt", output_dpath="data"):
    create_dir(output_dpath)
    urls = list(read_file(urls_fpath))
    # length of digits in an integer
    index_len = len(str(len(urls)))
                    
    error_urls = list()
    with tqdm.tqdm(total=len(urls)) as pbar:
        for i, url in enumerate(urls):
            url = url.strip()
            if not url:
                pbar.update(1)
                continue
                
            # Lấy tên file từ URL
            try:
                filename = url.split('/')[-1]
                if '?' in filename:
                    filename = filename.split('?')[0]
                if not filename.endswith('.txt'):
                    if '.' in filename:
                        base_name = filename.rsplit('.', 1)[0]
                        filename = f"{base_name}.txt"
                    else:
                        filename = f"{filename}.txt"
                
                output_fpath = os.path.join(output_dpath, filename)
            except:
                file_index = str(i+1).zfill(index_len)
                output_fpath = "".join([output_dpath, "/url_", file_index, ".txt"])
            
            is_success = write_content(url, output_fpath)
            if (not is_success):
                error_urls.append(url)
            pbar.update(1)

    return error_urls

def main(urls_fpath, output_dpath):
    crawl_urls(urls_fpath, output_dpath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VNExpress urls crawler")

    parser.add_argument("--input", 
                        default="urls.txt", 
                        help="urls txt file path",
                        dest="urls_fpath")
    parser.add_argument("--output", 
                        default="data", 
                        help="saved directory path",
                        dest="output_dpath")
    
    args = parser.parse_args()

    main(**vars(args))
