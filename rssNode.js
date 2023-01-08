let Parser = require('rss-parser');
let parser = new Parser();

(async () => {

  let feed = await parser.parseURL('https://partner-feeds.publishing.tamedia.ch/rss/24heures/la-une');
  console.log(feed.title);

  console.log(feed.items[10])

})();