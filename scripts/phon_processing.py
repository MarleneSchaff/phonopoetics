import argparse
import gzip
import json
import cmudict
import pronouncing as pron
import string

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("files_in", help = "gz.jsonl with preprocessed corpus")
	parser.add_argument("embed_out", help = "gz.jsonl with embeddings")

	args = parser.parse_args()

	phone_dict = cmudict.dict()
	entries = cmudict.entries()

	with gzip.open(args.files_in, "rt") as f_in, gzip.open(args.embed_out, "wt") as e_out:
		for line in f_in:
			jline = json.loads(line)
			print(jline["line"])
			phones = []
			stresses = []
			text = jline["line"].translate(str.maketrans("", "", string.punctuation))
			for word in text.split():
				print(word)
				try:
					try_phone = pron.phones_for_word(word)
#					phone_one = [try_phone[0]] if len(try_phone) > 1 else try_phone
					phone_one = try_phone[0]
					print(phone_one)
					phones.append(phone_one)
					#this is an oversimplification to automatically select the first pronunciation
					stresses.append(pron.stresses(phone_one))
					#pron.stresses_for_word(phones) produces list of possible stress patterns for a word
					#pron.syllable_count(phones) produces the number of syllables in a word (not useful though?)
				except: #KeyError:
					print("word not found in CMUdict")
			print(stresses)
			e_out.write(json.dumps(jline | {"phone_line": phones, "stresses": stresses}) + "\n")

#also need to somehow identify individual poems within gutenberg id which is for book?
