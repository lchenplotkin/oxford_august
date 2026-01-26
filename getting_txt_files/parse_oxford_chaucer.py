"""
<div2 id="actrade-9780192862921-div2-37" doi="10.1093/actrade/9780192862921.div2.37" role="poem">
<titleGroup id="actrade-9780192862921-titleGroup-248"><title><p alignment="centre"><?Page pageId="463"?><milestone num="463" unit="page" id="actrade-9780192862921-milestone-496"/><milestone unit="fragment-marker" id="actrade-9780192862921-milestone-1044"/>&#x201C;Against an Unconstant Woman&#x201D;</p></title>
</titleGroup>

<work id="actrade-9780192862921-work-8" doi="10.1093/actrade/9780192862921.work.8" role="poem">
<titleGroup id="actrade-9780192862921-titleGroup-216"><title><p alignment="centre"><?Page pageId="385"?><milestone num="385" unit="page" id="actrade-9780192862921-milestone-395"/><milestone unit="fragment-marker" id="actrade-9780192862921-milestone-1017"/>The Legends of Good Women</p></title></titleGroup>
<textMatter>

"""

# only using built-in Python libraries
import argparse
import json
import os
import os.path
import xml.etree.ElementTree as et
import logging

# set up a logger for each script is good practice, if a bit overkill.
logger = logging.getLogger("parse_oxford_chaucer")

# concatenate all the visible text under a given XML element.
def element_text(elem):
	return "".join([t for t in elem.itertext()])

# check whether this is being run *as* a script, in contrast to e.g. imported as a library
# (again, a bit overkill, but a good practice to internalize).
if __name__ == "__main__":

	# parse command-line arguments: think of the script as a whole as a function, this is
	# how to define its "signature".
	parser = argparse.ArgumentParser()
	parser.add_argument(dest="inputs", nargs="+", help="Any number of (but probably two) XML files constituting the Oxford Chaucer manuscript.")
	parser.add_argument("--log_level", dest="log_level", default="INFO", choices=logging.getLevelNamesMapping().keys())
	parser.add_argument("--json_output", dest="json_output", help="File name for saving JSON-formatted information extracted from manuscript.")
	parser.add_argument("--text_output_path", dest="text_output_path", help="Path for flat text files of each poem.")
	args = parser.parse_args()

	# use the command-line argument for the logging level to decide what information is printed while running.
	logging.basicConfig(level=logging.getLevelNamesMapping()[args.log_level])

	# keep track of how many lines of verse we process, just to print at the end as a sanity-check.
	line_count = 0

	# empty dictionary to populate with just the information we want to extract from the full XML.
	# as a minimal representation, each poem will just be a list of lists of strings (line groups
	# of lines, each of which is a string). 
	poems = {}

	for fname in args.inputs:
		
		logger.info("Processing file '%s'", fname)
		with open(fname, "rt") as ifd:

			# this is the actual XML parsing, everything else is just traversing the parsed structure.
			xml = et.fromstring(ifd.read())

			# three nested for-loops corresponding to poems/linegroups/lines. note how, at the poem-level,
			# there's a side quest to grab the title: other information could be gathered in similar ways,
			# particularly in concert with making the 'poems' structure a bit more expressive.
			for poem_element in xml.iterfind(".//work[@role='poem']"):
				poem = []
				title = element_text(poem_element.find("titleGroup/title/p"))
				print(title)
				logger.info("Processing poem '%s'", title)
				for linegroup_element in poem_element.iterfind(".//lineGroup"):
					linegroup = []
					for line_element in linegroup_element.iterfind(".//line"):
						line_count += 1
						linegroup.append(element_text(line_element))
					poem.append(linegroup)
				poems[title] = poem

	if args.json_output:
		logger.info("Writing %d lines from %d poems to '%s'", line_count, len(poems), args.json_output)
		with open(args.json_output, "wt") as ofd:
			ofd.write(json.dumps(poems, indent=4))

	if args.text_output_path:
		if not os.path.exists(args.text_output_path):
			os.makedirs(args.text_output_path)
		for title, linegroups in poems.items():
			ofname = os.path.join(args.text_output_path, "{}.txt".format(title))
			with open(ofname, "wt") as ofd:				
				logger.info("Writing '%s' to '%s'", title, ofname)
				ofd.write("\n\n".join(["\n".join(lg) for lg in linegroups]))
