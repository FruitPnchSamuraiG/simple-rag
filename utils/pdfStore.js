// import "pdf-parse"; // Peer dep
import { FaissStore } from "@langchain/community/vectorstores/faiss";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { OllamaEmbeddings } from "@langchain/community/embeddings/ollama";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

const embeddings = new OllamaEmbeddings({
  baseUrl: "http://127.0.0.1:11434",
  model: "all-minilm",
});
const loader = new PDFLoader('./pdf-test/example.pdf');
const docs = await loader.load();
const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 200,
});

const splits = await textSplitter.splitDocuments(docs);

const vectorstore = await FaissStore.fromDocuments(
  splits,
  embeddings
);

// save the vectorestore in an index file
vectorstore.save('./index') // directory name, specify a name specific to pdf's to maintain consistency