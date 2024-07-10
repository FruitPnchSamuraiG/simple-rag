import { createRetrievalChain } from "langchain/chains/retrieval";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { FaissStore } from "@langchain/community/vectorstores/faiss";
import { OllamaEmbeddings } from "@langchain/community/embeddings/ollama";
import { ChatOllama } from "@langchain/community/chat_models/ollama";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { PromptTemplate } from "@langchain/core/prompts";

// instantiating ollama model
const llm = new ChatOllama({
  baseUrl: "http://localhost:11434", // Default value
  model: "qwen2:1.5b", // Default value
});

// instantiating embedding model
const embeddings = new OllamaEmbeddings({
  baseUrl: "http://127.0.0.1:11434",
  model: "all-minilm",
});

// defining the prompt
const prompt = PromptTemplate.fromTemplate(
  "The question asked is {question} \n give the answer from the following context : {context}"
);

// load the index file
const vectorstore = await FaissStore.load("./", embeddings);

// load the retriever, this just does a similarity search nothing special
const retriever = vectorstore.asRetriever();

// storing the question
const question = "What is probate and letters of administration? Also give format of testimantory petition for probate"

// performing the similarity search
const context = await retriever.invoke(question);

const chain = await createStuffDocumentsChain({
  llm: llm,
  prompt,
  outputParser: new StringOutputParser(),
});

// invoke the chain, passing the question and context
console.log(await chain.invoke({question: question , context: context}));