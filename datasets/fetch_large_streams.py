"""Download the two large temporal streams used in E5 and verify their SHA-256.

The files come from the public collection by Min et al. (github.com/4AlexMin/dynamic-networks,
commit a1023fc6c54c70cd92d2152c8dc091829f7e2451), which redistributes SNAP's sx-superuser
and Network Repository's digg-friends in a uniform "u v t" format.
"""

import hashlib
import sys
import urllib.parse
import urllib.request
from pathlib import Path

COMMIT = "a1023fc6c54c70cd92d2152c8dc091829f7e2451"
BASE = f"https://raw.githubusercontent.com/4AlexMin/dynamic-networks/{COMMIT}/human/Interactive%20Digital%20Communities/"
FILES = {
    "sx-superuser.txt": "d99a66429d0c",
    "digg-friends.txt": "53a222c9ae10",
}


def main() -> int:
    here = Path(__file__).resolve().parent / "temporal"
    here.mkdir(exist_ok=True)
    checks = {}
    with open(here / "checksums.csv") as fh:
        for line in fh.read().splitlines()[1:]:
            parts = line.split(",")
            checks[parts[1]] = parts[-1]
    ok = True
    for name in FILES:
        target = here / name
        if not target.exists():
            print(f"downloading {name} ...")
            urllib.request.urlretrieve(BASE + urllib.parse.quote(name), target)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        good = digest == checks[name]
        ok &= good
        print(f"{name}: sha256 {'OK' if good else 'MISMATCH ' + digest}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
