import sys
import requests
import time
import openai
import tiktoken
import os
import tempfile

# Tools needed for preprocessing documents 
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Set the OpenAI API key
openai.api_key = 'KEY'

# Initialize langchain document loader
splitter = RecursiveCharacterTextSplitter(chunk_size=2046, chunk_overlap=100)

# Initialize tiktoken encoder
tokenizer = tiktoken.get_encoding('cl100k_base')

# Utility for retrieving a PDF from an URL
def retrieve_pdf_from_url(url):
	response = requests.get(url)
	with tempfile.NamedTemporaryFile(delete=False) as temp_file:
		temp_file.write(response.content)
		temp_file_path = temp_file.name
		temp_file_loader = PyMuPDFLoader(temp_file_path)
		temp_file_contents = temp_file_loader.load()
		temp_file.close()
		os.remove(temp_file_path)
		return temp_file_contents
	
# Utility for asking gpt-3.5-turbo a question
def get_response(txt):
	message = [{"role": "user", "content": txt}]
	response = openai.ChatCompletion.create(
						model = 'gpt-3.5-turbo',
						messages = message
					)
	return response["choices"][0]["message"]["content"] 

# SUMMARIZATION
#
# We will keep the summarization functionality very simple. To avoid
# feeding an entire paper into the API (which would take up too many
# tokens in most cases), we will do the following:
#
#  [1] Split the paper into chunks using langchain's recursive
#      splitter (RecursiveCharacterTextSplitter). See above for 
#      the parametrization.
#
#  [2] Extract the 3 most informative chunks and send them over to
#      OpenAI's API for summarization.
#
#  [3] Summarize the summaries into a single sentence in Japanese.
#      Hopefully, this will work most of the time, although occasionally
#      OpenAI's API behaves strangely. (A constitutional architecture
#      might be useful for this).
#
# There are many ways to do this, but this functions reasonably well, sort
# I decided to go for it. 
#
# The selection criteria for which chunks go into the summary are simple:
# I just take the three first chunks in the text. This will usually work
# well enough on scientific papers, since they begin with an abstract that
# summarize the work.
def document_get_summary(url):
	text = retrieve_pdf_from_url(url)			# raw text
	chunks = splitter.split_documents(text)		# chunked
	summaries = []
	
	for chunk in chunks:
		if len(summaries) >= 3:
			break
		else:
			try:
				response = get_response("Summarize the following article fragment as succinctly as possible\n" +
										"FRAGMENT: " + chunk.page_content + "\nSUMMARY: ")
				summaries.append(response)
			except:
				print("FAILURE")
			time.sleep(0.2)
	
	context = ''	# The "context" (i.e. the concatenation of summaries to be summarized itself)
	ctxlen = 0		# Keep track of the length of the context in tokens to prevent too long contexts
	for smr in summaries:
		if (ctxlen + len(tokenizer.encode(smr))) > 3000:
			break
		else:
			context += smr
			ctxlen += len(tokenizer.encode(smr))
	
	print (get_response('ARTICLE FRAGMENT: ' + context + '\nTLDR (in Japanese, 1 sentence only): '))
	
def main():
	if len(sys.argv) != 2:
		print("Command-line argument must be url to pdf")
		sys.exit(1)
	document_get_summary(sys.argv[1])
	
if __name__ == '__main__':
	main()
