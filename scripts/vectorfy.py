import argparse
import json
import gzip

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("phones_in", help = "gz.jsonl with lines tokenized into phonemes and stresses")
	parser.add_argument("data_out", help = "gz.jsonl with processed data")

	args = parser.parse_args()

	data = []
	with gzip.open(args.phones_in, "rt") as p_in:
		for line in p_in:
			jline = json.loads(line)
			data.append({"phones": jline["phone_line"], "stresses": jline["stresses"], "author": jline["author"]}) #also by title?

	#phones and stresses are each a list of strings

	phonemes = [d["phones"] for d in data]
	stress = [d["stresses"] for d in data]
	labels = [d["author"] for d in data]
	#so phonemes and stresses here are each a list of lists (well stresses = a list of lists of lists)

	unique_phones = set(phone for poem_line in phonemes for word in poem_line for phone in word.split())
	#^ set of each unique phoneme (some phonemes show french? irish? foreign?)
	phone_to_index = {phone: i for i, phone in enumerate(sorted(list(unique_phones)))}
	#^give each phoneme an index value

	phone_numbers = []
	for l in phonemes:
		new_line = []
		for w in l:
			new_word = [phone_to_index.get(item, item) for item in w.split()]
			new_line.append(new_word)
		phone_numbers.append(new_line)
#	print(phone_numbers)

	phone_flats = []
	for l in phone_numbers:
		new_l = [item for items in l for item in items]
		phone_flats.append(new_l)

	data_dict = {"labels": labels, "phones": phone_flats, "stresses": stress}

	print(data_dict)

	with gzip.open(args.data_out, "wt") as d_out:
		d_out.write(json.dumps(data_dict) + "\n")
