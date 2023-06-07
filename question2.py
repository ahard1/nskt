import sys
import requests
import time
import openai
import tiktoken
import os
import tempfile
import numpy as np
import sklearn

# Retrieve stuff we need for document preprocessing
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity

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

# Utility for asking for embeddings from text-embedding-ada-002
def get_embedding(txt):
	txt = txt.replace('\n', ' ')
	return openai.Embedding.create(input = txt, model="text-embedding-ada-002")['data'][0]['embedding']

############
# Database #
############

# For questions 2 and 3, we will use a retrieval model to build contexts for
# prompts. 

# Documents are stored as pieces ("chunks") in a database. When the user wants
# to ask a question, we get the embedding vector of the question, and then
# iterate over the document, comparing it to the embedding vectors of each chunk.

# We use the three closest chunks (token limits allowing) to build a "context"
# for the query, on the theory that embedding space closeness = relevance. The
# context is fed to the API alongside the query, ensuring that the AI will
# output an answer that draws only on information found in the paper it has 
# been fed.

database = [] # This can be a simple list for question 2, as the system 
			  # only has to handle a single document at a time (no "switching")

class Chunk:
	def __init__(self, text, vector):
		self.text = text
		self.vector = vector

# commits the document to database, chunking and embedding it along the way
def document_memorize(url):
	text = retrieve_pdf_from_url(url)
	textchunks = splitter.split_documents(text)
	for textchunk in textchunks:
		chunk = Chunk(textchunk.page_content, get_embedding(textchunk.page_content))
		database.append(chunk)
		time.sleep(0.2)

# We use the same summarization as for question1, but modify it slightly since
# the documents are now placed in the database
def summarize():
	summaries = []
	for chunk in database:
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

# For question-answering, we build a context from chunks that are close in vector-space
# to the user's question
def answer(query):
	qvec = get_embedding(query) # Get the embedding vector for the query
	qvec = np.array(qvec)
	qvec = qvec.reshape(1, -1)
	out = []
	
	# Calculate the distance between the query and each chunk
	for chunk in database:
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

	
def main():
	global api_calls
	
	if len(sys.argv) != 2:
		print("Command-line argument must be URL to pdf")
		sys.exit(1)

	# Build the document database
	document_memorize(sys.argv[1]) 
	
	# Post a one-sentence summary of the article
	summarize()
	
	# Run the chatbot loop
	while True:
		user_input = input("> ")
		if user_input.lower() == 'exit' or user_input.lower() == 'quit' or user_input.lower() == 'q':
			break
		else:
			answer(user_input) 
	
	# We're exiting, so clear out the downloaded documents
	print("EXITING")

if __name__ == '__main__':
	main()
