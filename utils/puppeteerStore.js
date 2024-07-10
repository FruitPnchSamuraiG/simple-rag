import { OllamaEmbeddings } from "@langchain/community/embeddings/ollama";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { load } from "cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { ChatOllama } from "@langchain/community/chat_models/ollama";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { PromptTemplate } from "@langchain/core/prompts";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { PuppeteerWebBaseLoader } from "@langchain/community/document_loaders/web/puppeteer";
import natural from "natural";
import { removeStopwords } from "stopword";
import { FaissStore } from "@langchain/community/vectorstores/faiss";

const tokenizer = new natural.WordTokenizer();
const stemmer = natural.PorterStemmer;

// Preprocessing function to clean and filter text
const preprocessText = (text) => {
  return text.replace(/\s+/g, " ").trim();

  // // Tokenize text
  // let tokens = tokenizer.tokenize(text.toLowerCase());

  // // Remove stopwords
  // tokens = removeStopwords(tokens);

  // // Perform stemming
  // tokens = tokens.map((token) => stemmer.stem(token));

  // // Join tokens back into a single string
  // return tokens.join(" ");
};

// Function to scrape and clean content
const scrapeContent = async (url) => {
  const loader = new PuppeteerWebBaseLoader(url, {
    launchOptions: { headless: true },
    gotoOptions: { waitUntil: "domcontentloaded" },
    async evaluate(page, browser) {
      const html = await page.content();
      const $ = load(html);

      // Extract and clean text from relevant tags
      let textElements = $("div, p, span, li, a")
        .map((i, el) => preprocessText($(el).text()))
        .get()
        .join(" ");

      await browser.close();
      return textElements;
    },
  });

  return loader.load();
};

// Scrape content from the given URL
const docs = await scrapeContent(
  "https://en.wikipedia.org/wiki/Sekiro:_Shadows_Die_Twice"
);
console.log(docs);

// generating embeddings for the same
const embeddings = new OllamaEmbeddings({
  baseUrl: "http://127.0.0.1:11434",
  model: "all-minilm",
});


// splitting it into relevant chunks
const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 200,
  chunkOverlap: 50,
});

const allSplits = await textSplitter.splitDocuments(docs);

// load the vector sotore
const vectorstore = await FaissStore.fromDocuments(
  allSplits,
  embeddings
);

// save the vectorestore in an index file
vectorstore.save('./index') // directory name, specify a name specific to pdf's to maintain consistency
