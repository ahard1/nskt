import sys
import requests
import time
import openai
import tiktoken
import sklearn		# we use sklearn to measure the similarity of embedding vectors
import numpy as np	# ditto for numpy
import tempfile
import os
import re			# regular expressions for extracting urls

# Retrieve stuff we need for document preprocessing
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity

# Set the OpenAI API key
openai.api_key = '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'

# Initialize tiktoken encoder
tokenizer = tiktoken.get_encoding('cl100k_base')

# Initialize langchain document loader
splitter = RecursiveCharacterTextSplitter(chunk_size=2046, chunk_overlap=100)

# Temporary file path
temppath = 'C:/Users/arthu/Downloads/junktest.pdf'
current_pdf = ''

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

# Utility for asking for embeddings from text-embedding-ada-002
def get_embedding(txt):
	txt = txt.replace('\n', ' ')
	return openai.Embedding.create(input = txt, model="text-embedding-ada-002")['data'][0]['embedding']

# The database needs to be able to hold multiple articles
database = {}

class Chunk:
	def __init__(self, text, vector):
		self.text = text
		self.vector = vector

def document_has_been_added(url):
	return database.get(url) is not None

# commits the document to database, chunking and embedding it along the way
def document_memorize(url):
	if document_has_been_added(url):
		return
	else:
		text = retrieve_pdf_from_url(url)
		textchunks = splitter.split_documents(text)
		data = []
		for textchunk in textchunks:
			chunk = Chunk(textchunk.page_content, get_embedding(textchunk.page_content))
			data.append(chunk)
			time.sleep(0.2)
		database[url] = data

# Summarize		
def summarize(current):
	summaries = []
	chunks = database.get(current)
	
	if chunks == None:
		return False
	
	for chunk in chunks:
		if len(summaries) >= 3:
			break
		else:
			try:
				response = get_response("Summarize the following article fragment as succinctly as possible\n" +
										"FRAGMENT: " + chunk.text + "\nSummary: ")
				summaries.append(response)
			except:
				print("FAILURE")
			time.sleep(0.2)
	
	context = ''
	ctxlen = 0
	for smr in summaries: 
		if (ctxlen + len(tokenizer.encode(smr))) > 3000:
			break
		else:
			context += smr
			ctxlen += len(tokenizer.encode(smr))

	print (get_response('ARTICLE FRAGMENT: ' + context + '\nTLDR (in Japanese, 1 sentence only): '))
	
def answer(query, current):
	qvec = get_embedding(query) # Get the embedding vector for the query
	qvec = np.array(qvec)
	qvec = qvec.reshape(1, -1)
	out = []
	
	# Calculate the distance between the query and each chunk
	chunks = database.get(current)
	
	if chunks == None:
		return False
	
	for chunk in chunks:
		cvec = np.array(chunk.vector)
		cvec = cvec.reshape(1, -1)
		out.append([chunk, cosine_similarity(qvec, cvec)])
		
	# Sort the documents by their closeness to the query vector
	out.sort(reverse=True, key=lambda x: x[1][0][0]) 
	
	# Construct the context, which should ideally consist of three
	# retrieved documents, and be less than 3000 tokens in length.
	# Note that with this implementation, the length of the context
	# can vary widely.
	maxlen = 3000
	counter = 0
	ctxlen = 0
	context = ''
	for chunkandvec in out:
		chunk = chunkandvec[0]
		if (ctxlen + len(tokenizer.encode(chunk.text))) > maxlen:
			break
		elif counter >= 3:
			break
		else:
			context += chunk.text + ' ' 
			ctxlen += len(tokenizer.encode(chunk.text))
			counter += 1
	
	# Construct the query and fire it off to the OpenAI API
	reply = get_response('This is a collection of fragments of text from an article. Use the information ' +
						'in it to answer the question. If you cannot answer the question, say so. Do not ' +
						'make anything up.\nFRAGMENTS: ' + context + '\nQUESTION: ' + query + '\nANSWER IN JAPANESE: ')
						
	print(reply)
	
def extract_url(str):
	url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
	urls = re.findall(url_pattern, str)
	
	if len(urls) > 0:
		for url in urls:
			if url.endswith('pdf'):
				return url
	
	return None
	
def main():
	if len(sys.argv) != 2:
		print("Command-line argument must be URL to pdf")
		sys.exit(1)

	# Keep track of current argument
	current = sys.argv[1]
	
	# Initialize
	document_memorize(current) 
	
	# Summarize
	summarize(current) 
	
	# Run the chatbot loop
	while True:
		user_input = input("> ")
		newpdf = extract_url(user_input)
		if newpdf == None or newpdf == current:
			if user_input.lower() == 'exit' or user_input.lower() == 'quit' or user_input.lower() == 'q':
				break
			else:
				answer(user_input, current)
		elif document_has_been_added(newpdf):
			current = newpdf
			answer(user_input, current)
		else:
			print("NEWPDF: " + newpdf)
			current = newpdf
			document_memorize(current)
			summarize(current)
			
	print("EXITING")

if __name__ == '__main__':
	main()
	