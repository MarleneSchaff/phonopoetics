import argparse
import json
import gzip
import csv
import pandas as pd

if __name__== "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("corpus_in", help = "gz.jsonl containing texts with text ids")
	parser.add_argument("metadata_in", help = "csv containing gutenberg text metadata")
	parser.add_argument("corpmeta_out", help = "gz.jsonl containing corpus with metadata")
	args = parser.parse_args()

	df = pd.read_csv(args.metadata_in)

	with open(args.corpus_in, "rt") as text_in, gzip.open(args.corpmeta_out, "wt") as cm_out:
		for line in text_in:
			jline = json.loads(line)
			id_num = int(jline["gid"])
			author = df.loc[df["Text#"] == id_num, "Authors"].item()
			title = df.loc[df["Text#"] == id_num, "Title"].item()
			cm_out.write(json.dumps({"line": jline["s"], "g_id": jline["gid"], "title": title, "author": author}) + "\n")

#also include ["Language"] to filter for en (English)?
