import { OllamaEmbeddings } from "@langchain/community/embeddings/ollama";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import "cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { ChatOllama } from "@langchain/community/chat_models/ollama";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { PromptTemplate } from "@langchain/core/prompts";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { PuppeteerWebBaseLoader } from "langchain/document_loaders/web/puppeteer";

// initializing ollama model
const ollama = new ChatOllama({
  baseUrl: "http://localhost:11434", // Default value
  model: "qwen2:1.5b", // Default value
});

// scraping the web
const loader = new CheerioWebBaseLoader("https://stackoverflow.com/questions/50200597/how-to-update-a-file-placed-in-my-github-repository-using-command-line", {
  selector: "p, h1, h2, h3, h4, h5, h6, li, span, div, a, blockquote, td, th", 
});

// loading the scraped docs
let docs = await loader.load();
console.log(docs);

// // splitting it into relevant chunks
// const textSplitter = new RecursiveCharacterTextSplitter({
//   chunkSize: 200,
//   chunkOverlap: 50,
// });
// const allSplits = await textSplitter.splitDocuments(docs);

// // generating embeddings for the same
// const embeddings = new OllamaEmbeddings({
//   baseUrl: "http://127.0.0.1:11434",
//   model: "mxbai-embed-large",
// });
// const vectorStore = await MemoryVectorStore.fromDocuments(
//   allSplits,
//   embeddings
// );

// // prompt template
// const prompt = PromptTemplate.fromTemplate(
//   "The question asked is {question} \n give the answer from the following context: {context}"
// );

// // defining the chain
// const chain = await createStuffDocumentsChain({
//   llm: ollama,
//   outputParser: new StringOutputParser(),
//   prompt,
// });

// const question = "What is this?";
// const searchResults = await vectorStore.similaritySearch(question);
// console.log(
//   await chain.invoke({
//     question: question,
//     context: searchResults,
//   })
// );
