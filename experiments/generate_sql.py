import random
import datetime
import pathlib

random.seed(42)
mods = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta"]
rows = []
start = datetime.datetime(2024, 6, 1)

for i in range(500):
    mod = random.choice(mods)
    failures = random.randint(1, 25)
    mtbf = random.randint(50, 600)
    runtime = random.randint(mtbf // 2, mtbf * 3)
    ts = start + datetime.timedelta(hours=i * 3)
    rows.append((mod, failures, mtbf, runtime, ts.strftime("%Y-%m-%d %H:%M:%S")))

sql_lines = [
    "CREATE DATABASE IF NOT EXISTS reliability_test DEFAULT CHARSET utf8mb4;",
    "USE reliability_test;",
    "DROP TABLE IF EXISTS reliability_records;",
    "CREATE TABLE reliability_records (id INT AUTO_INCREMENT PRIMARY KEY, module VARCHAR(64) NOT NULL, failures INT NOT NULL, mtbf DOUBLE NOT NULL, runtime DOUBLE NOT NULL, ts TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP);",
]
insert_head = "INSERT INTO reliability_records (module, failures, mtbf, runtime, ts) VALUES"
values_str = ",\n".join([f"('{m}', {f}, {mt}, {r}, '{t}')" for m, f, mt, r, t in rows])
sql_lines.append(insert_head)
sql_lines.append(values_str + ";")

path = pathlib.Path("experiments/reliability_test.sql")
path.parent.mkdir(exist_ok=True)
path.write_text("\n".join(sql_lines), encoding="utf-8")
print("Generated SQL with", len(rows), "rows ->", path.resolve())
