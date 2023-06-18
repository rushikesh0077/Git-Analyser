import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from re import match
import urllib.request
from langchain.document_loaders import NotebookLoader
import os
import openai
from langchain.chat_models import ChatOpenAI
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQAWithSourcesChain
from dotenv import load_dotenv
load_dotenv()


class repo_analyser:
    def __init__(self,user_url) -> None:
        self.user_url = user_url
        self.repo_links = self.repo_link_extractor()
        self.raw_code_links = self.code_link_extractor()
        self.notebook_downloader()
        self.document = self.notebook_loader()
        self.docsearch = self.splitter_and_embeddings()

    def repo_link_extractor(self):
        headers = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"}
        reqs = requests.get(self.user_url,headers = headers).text
        soup = BeautifulSoup(reqs, 'html.parser')
        url_lst = []
        for link in soup.find_all('a'):
            link_1 = link.get('href')
            url_lst.append(urljoin(self.user_url,link_1))
        repo_link = list(set(list(filter(lambda v: match('(^https)(.*tab=repositories$)', v), url_lst))))[0]
        reqs = requests.get(repo_link,headers = headers).text
        soup = BeautifulSoup(reqs, 'html.parser')
        repos = []
        for link in soup.find_all('a'):
            if link.get('itemprop') == 'name codeRepository':
                link_1 = link.get('href')
                repos.append(urljoin(self.user_url,link_1))
        return repos
    
    def code_link_extractor(self):
        headers = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"}
        all_links = []
        for link in self.repo_links:
            reqs = requests.get(link,headers = headers).text
            soup = BeautifulSoup(reqs, 'html.parser')
            for l in soup.find_all('a'):
                link_1 = l.get('href')
                all_links.append(urljoin(link,link_1))
        code_link = list(set(list(filter(lambda v: match('(^https)(.*ipynb$)', v), all_links))))
        raw_code_links = []
        for code_link in code_link:
            reqs = requests.get(code_link,headers = headers).text
            soup = BeautifulSoup(reqs, 'html.parser')
            for l in soup.find_all('a'):
                link_1 = l.get('href')
                raw_code_links.append(urljoin(link,link_1))
        final_raw_code_links = list(set([link for link in raw_code_links if 'raw' in link]))
        return final_raw_code_links

    def notebook_downloader(self):
        for link in self.raw_code_links:
            urllib.request.urlretrieve(link,os.path.join('codes',link.split('/')[-1]))

    def notebook_loader(self):
        doc = []
        for link in self.raw_code_links:
            loader = NotebookLoader(os.path.join('codes',link.split('/')[-1]))
            data = loader.load()
            doc.extend(data)
        return doc
    
    def splitter_and_embeddings(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size = 2056, chunk_overlap = 32)
        splitted_doc = splitter.split_documents(self.document)
        embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])
        docsearch = FAISS.from_documents(splitted_doc, embeddings)
        return docsearch
    
    def final_answer(self):
        query = f"""Act as a programming expert. Your job is to find out the output for given python codes. Find out the one most technically complex code and its source and give the the link for the code from links: {self.raw_code_links} and justification why the code is technically complex but dont add any code in your justification.
            Always give the answer in below format:
            Justification: The following repository [justification]...
            code link: link"""
        llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo-0613')
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=self.docsearch.as_retriever())
        result = chain({"question": query})
        ans = result['answer']
        justitication = ans.split('\n')[0]
        link = ans.split('\n')[2]
        return justitication,link