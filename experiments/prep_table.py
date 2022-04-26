table_str = """
Class\tPrecision (%)
black-king black-queen black-rook black-bishop black-knight black-pawn white-king white-queen white-rook white-bishop white-knight white-pawn
100.0 92.9 99.0 95.9 99.8 96.6 99.6 98.6 95.3 98.2 99.7 94.1
"""

table_values = [el for el in table_str.split("\n") if el]
print(table_values)

tbl_hdrs = table_values[0].split("\t")
print(tbl_hdrs)
table_header = "| " + " | ".join(tbl_hdrs) + " |" + "\n| " + "-" * len(tbl_hdrs[0]) + " | " + "-" * len(tbl_hdrs[1]) + " |"

table_rows = ""
tbl_rows = table_values[1].split()
ap_values = table_values[2].split()
for i in range(len(tbl_rows)):
    table_rows += "\n| " + tbl_rows[i] + " | " + ap_values[i] + " |"

print(table_header + table_rows)
