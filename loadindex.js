import { FaissStore } from "@langchain/community/vectorstores/faiss";
import { OllamaEmbeddings } from "@langchain/community/embeddings/ollama";
import { ChatOllama } from "@langchain/community/chat_models/ollama";
import { PromptTemplate } from "@langchain/core/prompts";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { StringOutputParser } from "@langchain/core/output_parsers";

// initializing ollama model
const ollama = new ChatOllama({
  baseUrl: "http://localhost:11434", // Default value
  model: "qwen2:1.5b", // Default value
});

const embeddings = new OllamaEmbeddings({
  baseUrl: "http://127.0.0.1:11434",
  model: "all-minilm",
});

const vectorstore = await FaissStore.load('./', embeddings)

// prompt template
const prompt = PromptTemplate.fromTemplate(
  "The question asked is {question} \n give the answer from the following context and your intelligence : {context}"
);

// defining the chain
const chain = await createStuffDocumentsChain({
  llm: ollama,
  outputParser: new StringOutputParser(),
  prompt,
});

// storing the question we want to ask
const question = "What is probate and letters of administration? Also give format of testimantory petition for probate";

// performing similarity search for getting the context
const similaritySearch = await vectorstore.similaritySearch(question);

// printing the page numbers and line numbers
similaritySearch.forEach((val, index)=>{
  console.log(val.metadata.loc)
})

// invoking/running the chain by passing the question and context obtained from similarity search
let response = await chain.invoke({question: question, context: similaritySearch })
response = response + "\n"
console.log(response); // logging/printing the response