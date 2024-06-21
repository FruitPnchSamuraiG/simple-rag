import { ChatOllama } from "@langchain/community/chat_models/ollama";
import { DuckDuckGoSearch } from "@langchain/community/tools/duckduckgo_search";
import { webScraper } from "./utils/web-scraper.js";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OllamaEmbeddings } from "@langchain/community/embeddings/ollama";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { PromptTemplate } from "@langchain/core/prompts";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { StringOutputParser } from "@langchain/core/output_parsers";

// initializing ollama model
const ollama = new ChatOllama({
  baseUrl: "http://localhost:11434", // Default value
  model: "qwen2:1.5b", // Default value
  temperature: 0.2,
});

const question = "Why did BJP lose seats in Ayodhya?";

const tool = new DuckDuckGoSearch({ maxResults: 10 });

let searchResults = await tool.invoke(question);

// parsing the reuslts to json so they can be formatted to what langchain requires
searchResults = JSON.parse(searchResults);

// extracting link 
const firstLink = searchResults[0].link

// doesn't the name sound super cool?
// it's not that cool in realit it just joins the top result with the rest of the snippets
const alphaSearch = async () => {
  const search = searchResults.map((val) => {
    return {
      pageContent: JSON.stringify({ title: val.title, snippet: val.snippet }),
      metadata: { source: val.link },
    };
  });
  const webScrape = await webScraper(firstLink);
  return search.concat(webScrape)
};

const docs = await alphaSearch()


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

// this vector store actually stores in the cache memory
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

let response = await chain.invoke({
  question: question,
  context: similaritySearch,
});

console.log(response);
