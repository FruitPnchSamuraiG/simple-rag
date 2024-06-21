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
});

const tool = new DuckDuckGoSearch({ maxResults: 10 });

let searchResults = await tool.invoke(
  "What did apple say WWDC 2024?"
);

searchResults = JSON.parse(searchResults)
console.log(searchResults);
const docs = searchResults.map((val)=>{
  return {
    pageContent: JSON.stringify({title: val.title, snippet: val.snippet}),
    metadata: {source: val.link}
  }
})


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
let vectorStore;
try{
  vectorStore = await MemoryVectorStore.fromDocuments(
    allSplits,
    embeddings
  );
} catch(error){
  console.log(error);
}


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
const question = "Explain";
const similaritySearch = await vectorStore.similaritySearch(question);
console.log("Search Results: ",similaritySearch);
let response = await chain.invoke({question: question, context: similaritySearch })
response = response + "\n"
for(let i = 0 ; i < similaritySearch.length ; i++){
  response = response + `[${i}](${similaritySearch[i].metadata.source}) `
}
console.log(response);