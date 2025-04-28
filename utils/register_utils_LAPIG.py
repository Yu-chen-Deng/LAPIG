import os
import site

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    site_packages_dir = site.getsitepackages()[0]
    pth_file_path = os.path.join(site_packages_dir, 'utils_LAPIG.pth')
    with open(pth_file_path, 'w') as f:
        f.write(current_dir + '\n')
    print(f"Already add {current_dir} into {pth_file_path}")
    
if __name__ == "__main__":
    main()
    