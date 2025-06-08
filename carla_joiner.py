import glob
import re


def join_files(output_filename, part_pattern):
    # Find all part files and sort them in order
    part_files = sorted(
        glob.glob(part_pattern), key=lambda x: int(re.search(r"(\d+)$", x).group(1))
    )
    print("Joining files:", part_files)
    with open(output_filename, "wb") as outfile:
        for part in part_files:
            with open(part, "rb") as infile:
                outfile.write(infile.read())
    print(f"Created {output_filename}")


# Usage
join_files("m_joined.bin", "m.bin.part*")

# https://drive.iust.ac.ir/index.php/s/yF97RP3D7WfdDZ5
# https://drive.iust.ac.ir/index.php/s/d46wkSHaS87oTzw
# https://drive.iust.ac.ir/index.php/s/q8RaBGYsAmjFP8X
