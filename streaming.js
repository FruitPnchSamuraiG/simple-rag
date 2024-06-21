import { ChatOllama } from "@langchain/community/chat_models/ollama";
import { OllamaEmbeddings } from "@langchain/community/embeddings/ollama";
import { DuckDuckGoSearch } from "@langchain/community/tools/duckduckgo_search";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { PromptTemplate } from "@langchain/core/prompts";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { StringOutputParser } from "@langchain/core/output_parsers";

// initializing ollama model
const ollama = new ChatOllama({
  baseUrl: "http://localhost:11434", // Default value
  model: "qwen2:1.5b", // Default value
  temperature: 0.2
});

const question = "How many goals did cristiano score last night?";

const tool = new DuckDuckGoSearch({ maxResults: 10 });

let searchResults = await tool.invoke(question);

searchResults = JSON.parse(searchResults);

const docs = searchResults.map((val) => {
  return {
    pageContent: JSON.stringify({ title: val.title, snippet: val.snippet }),
    metadata: { source: val.link },
  };
});

// splitting it into relevant chunks
const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 200,
  chunkOverlap: 50,
});
const allSplits = await textSplitter.splitDocuments(docs);

// generating embeddings for the same
const embeddings = new OllamaEmbeddings({
  baseUrl: "http://127.0.0.1:11434",
  model: "mxbai-embed-large",
});
const vectorStore = await MemoryVectorStore.fromDocuments(
  allSplits,
  embeddings
);

// prompt template
const prompt = PromptTemplate.fromTemplate(
  "The question asked is {question} \n give the answer from the following context : {context}"
);

// defining the chain
const chain = await createStuffDocumentsChain({
  llm: ollama,
  outputParser: new StringOutputParser(),
  prompt,
});

// Ask the question
const similaritySearch = await vectorStore.similaritySearch(question);
console.log("Search Results: ", similaritySearch);

let response = await chain.stream({
  question: question,
  context: similaritySearch,
});
let string = ""
for await (const part of response) {
  string += part
  console.log(string);
}

response = response + "\n"
for(let i = 0 ; i < similaritySearch.length ; i++){
  response = response + `[${i+1}](${similaritySearch[i].metadata.source}) `
}
console.log(response);