tolerance = 100
if matched == "DIFF":
	riv_tags = list(tags)
	tags = []
	riv_words_normed = []
	for word in riv_words:
		riv_words_normed.append(norm_word(word))

	# Track which Riverside indices have been used
	used_indices = []

	for i, word in enumerate(ox_words):
		ox_clean = word.lower().strip()
		
		# Check if this is a naught/not conversion case
		if flag == 'green':
			riv_word_clean = riv_words[i].lower().strip() if i < len(riv_words) else ''
			
			# Riverside has naught/nought, Oxford has not/nat
			if riv_word_clean in {'naught', 'nought'} and ox_clean in {'not', 'nat'}:
				# Tag Oxford word with "not"
				tags.append('{*not@adv*}')
				continue
			# Riverside has not/nat, Oxford has naught/nought
			elif riv_word_clean in {'not', 'nat'} and ox_clean in {'naught', 'nought'}:
				# Tag Oxford word with "nought"
				tags.append('{*nought@adv*}')
				continue
		
		# Default behavior: match by normalized word
		normed_ox_word = norm_word(word)
		if normed_ox_word in riv_words_normed:
			# Find the first unused occurrence of this word
			found_index = None
			for j, riv_normed in enumerate(riv_words_normed):
				if riv_normed == normed_ox_word and j not in used_indices:
					found_index = j
					used_indices.append(j)
					break
			
			if found_index is not None:
				tags.append(riv_tags[found_index])
			else:
				# All occurrences used, append empty
				tags.append('')
		else:
			#HERE, look through the oxford_prelim.json
			#if the entry for ox_clean has an option that is at least tolerance times more likely than any other option (or if there is only one option):
				# tag the oxford word with that option
			else: 
				flag = ""
				tags.append('')
if flag == "yellow":
	flag = "green"

