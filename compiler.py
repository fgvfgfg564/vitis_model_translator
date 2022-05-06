import os
import sys
import argparse

archfile = "/home/xyhang/arch.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=str, help="directory to compile")

    args = parser.parse_args()
    folder = args.dir

    if not os.path.isfile(archfile):
        raise FileNotFoundError(f"Arch file not found: {archfile}")

    filelist = os.listdir(folder)
    for filename in filelist:
        name, ext = os.path.splitext(filename)
        if ext == ".h5":
            optFile = name + ".opt"
            if optFile not in filelist:
                raise FileNotFoundError(
                    f"Input information file : '{optFile}' not found in current directory"
                )
            with open(os.path.join(folder, optFile), "r") as f:
                options = f.read()
            moduleName = os.path.split(name)[-1]
            cmd = f"vai_c_tensorflow2 -m {os.path.join(folder, filename)} -n {moduleName} -o {folder} -a {archfile} {options}"
            os.system(cmd)
