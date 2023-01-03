from pathlib import Path

query_path = Path(str(Path(__file__).resolve().parents[2]) + "/queries")

def read_query(file_name: str) -> str:
    """
    Read sql query and return it as a string
    """
    path = query_path / file_name
    with open(path) as f:
        query = f.read()

    return query
