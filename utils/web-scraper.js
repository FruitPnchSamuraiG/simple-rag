import { PuppeteerWebBaseLoader } from "@langchain/community/document_loaders/web/puppeteer";
import { load } from 'cheerio';

export async function webScraper(url) {
  // scraping the web
  const loader = new PuppeteerWebBaseLoader(url, {
    launchOptions: {
      headless: true,
    },
    gotoOptions: {
      waitUntil: "domcontentloaded",
    },
    /**  Pass custom evaluate , in this case you get page and browser instances */
    async evaluate(page, browser) {
      const html = await page.content();
      const $ = load(html);
      const textElements = $("div,p, h1, h2, h3, h4, h5, h6, ul, ol,li").text();
      await browser.close();
      return textElements;
    },
  });

  // loading the scraped docs
  return await loader.load();
}
