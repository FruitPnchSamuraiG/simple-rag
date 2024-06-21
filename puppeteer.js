import { OllamaEmbeddings } from "@langchain/community/embeddings/ollama";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { load } from 'cheerio';
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { ChatOllama } from "@langchain/community/chat_models/ollama";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { PromptTemplate } from "@langchain/core/prompts";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { PuppeteerWebBaseLoader } from "@langchain/community/document_loaders/web/puppeteer";

// initializing ollama model
const ollama = new ChatOllama({
  baseUrl: "http://localhost:11434", // Default value
  model: "qwen2:1.5b", // Default value
});

// scraping the web
const loader = new PuppeteerWebBaseLoader("https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/ArrayBuffer", {
  launchOptions: {
    headless: false,
  },
  gotoOptions: {
    waitUntil: "domcontentloaded",
  },
  /**  Pass custom evaluate , in this case you get page and browser instances */
  async evaluate(page, browser) {
    const html = await page.content();
    const $ = load(html);
    const textElements = $('div,p, h1, h2, h3, h4, h5, h6, ul, ol,li').text();
    await browser.close();
    return textElements;
  },
});

// loading the scraped docs
let docs = await loader.load();
console.log(docs);

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
  "Summarize the following: {context}"
);

// defining the chain
const chain = await createStuffDocumentsChain({
  llm: ollama,
  outputParser: new StringOutputParser(),
  prompt,
});

const question = "Explain ArrayBuffers";
const searchResults = await vectorStore.similaritySearch(question);
console.log("Search Results: ",searchResults);
console.log(
  await chain.invoke({
    question: question,
    context: searchResults,
  })
);
