# Dataset provenance

All data used in the evaluation are public benchmark or research datasets. No personal data beyond what the original publishers released in anonymised form is used, and no data were generated to stand in for real data: generated graphs appear only in the unit tests (known-answer oracles), the parameter pilot (tuning only) and the scalability experiment E5b (to control the number of vertices), and are labelled as synthetic wherever they appear.

SHA-256 checksums of every file are in `dimacs/checksums.csv` and `temporal/checksums.csv`.

## DIMACS graph colouring instances (`dimacs/`)

* **Origin:** the graph colouring instances of the Second DIMACS Implementation Challenge (Johnson and Trick, 1996), as listed on Michael Trick's "Graph Coloring Instances" page.
* **Copy used:** GitHub repository `BartMassey/instances`, commit `83b4e11b5d83bcf44feb628d1e4f7a994de0a0cf` (the official FTP/HTTP download was not reachable from the build machine's network policy).
* **Checks:** for every instance the vertex count and the edge count in the file were compared with Trick's page (`dimacs/trick_table.csv`, extracted on 3 October 2026). All matched except two apparent typing errors on the page: DSJC250.5 (page 31,366, file 31,336 edge lines) and DSJC500.5 (page 125,249, file 125,248). Several files list every edge twice; `graph_loader.read_dimacs` merges duplicates, so `unique_edges` in `checksums.csv` is the number of distinct undirected edges (for example 15,668 for DSJC250.5 and 1,980 for queen11_11).
* **Chromatic numbers** quoted in the report are those printed on Trick's page; where the page shows "?", no value is quoted.

## Temporal interaction streams (`temporal/`)

* **Copy used:** GitHub repository `4AlexMin/dynamic-networks`, commit `a1023fc6c54c70cd92d2152c8dc091829f7e2451`, which redistributes public temporal networks in a uniform `u v t` format (Min, Shang, Liu and Chen, 2025).

| File | Original source | Original reference | Check performed |
|---|---|---|---|
| `primary-school.txt` | SocioPatterns | Stehlé et al. (2011) | 242 vertices and 125,773 20-second contact records; merging consecutive records of a pair gives 77,521 contact events against 77,602 reported in the paper (0.1% fewer, probably cleaning differences) |
| `high-school-2013.txt` | SocioPatterns | Mastrandrea, Fournet and Barrat (2015) | 327 vertices, 188,508 records |
| `SFHH-conf.txt` | SocioPatterns | Génois and Barrat (2018) | 403 vertices, 70,261 records |
| `CollegeMsg.txt` | SNAP | Panzarasa, Opsahl and Carley (2009) | 1,899 vertices and 59,835 temporal edges, equal to the SNAP page |
| `email-Eu-core-temporal.txt` | SNAP | Paranjape, Benson and Leskovec (2017) | 986 vertices and 332,334 temporal edges, equal to the SNAP page |
| `sx-superuser.txt` | SNAP | Paranjape, Benson and Leskovec (2017) | 194,085 vertices, 1,443,339 records (as listed by the collection) |
| `digg-friends.txt` | Network Repository | see the collection's README | 279,374 vertices, 1,729,983 records (as listed by the collection) |

A copy of CollegeMsg from a different GitHub mirror was rejected during preparation because its counts (1,769 vertices) did not match SNAP.

**Licences.** SocioPatterns datasets are released under a Creative Commons Attribution-NonCommercial licence; the SNAP and Network Repository datasets are distributed for research use. Check the original pages before any redistribution. The two large files are excluded from the repository by `.gitignore`; `fetch_large_streams.py` downloads them from the pinned commit and verifies their checksums.

## How the streams become dynamic graphs

`workloads.sliding_window` keeps edge {u, v} present while the pair has interacted within the last W seconds (W = 1 hour for face-to-face contacts, 1 week for message and e-mail networks). A first interaction inserts the edge; when W seconds pass without a new interaction the edge is deleted. Self-interactions are ignored. The vertex set is fixed (all vertices of the file), and the graph starts empty.
