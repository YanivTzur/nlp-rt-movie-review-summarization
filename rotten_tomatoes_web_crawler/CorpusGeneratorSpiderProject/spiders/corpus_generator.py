import scrapy
import re
import json

dvd_streaming_all_url = 'https://www.rottentomatoes.com/api/private/v2.0/browse?maxTomato='\
                          +\
                          '100&maxPopcorn=100&services=amazon%3Bhbo_go%3Bitunes%3Bnetflix_iw%3Bvudu%3Bamazon_prime'\
                          +\
                          '%3Bfandango_now&certified&sortBy=release&type=dvd-streaming-all&page=1'
MAX_PAGE_NUM = 312  # The last page to retrieve, currently decided manually by trial and error.


class CorpusGenerator(scrapy.Spider):
    name = 'corpus_generator'

    start_urls = [dvd_streaming_all_url]
    id = 1
    pageNum = 1

    def parse(self, response):
        def extract_with_css(query):
            return response.css(query).extract_first().strip()

        def extract_name():
            name = response.css("h1.title.hidden-xs::text")
            if len(name) > 0:
                return extract_with_css("h1.title.hidden-xs::text")
            return response.css("#movie-title::text").extract_first().strip()

        def extract_summary():
            consensus_element = response.css("p.critic_consensus.superPageFontColor").extract_first()
            end_of_first_span = consensus_element.find("</span>") + len("</span>")
            consensus_element = consensus_element[end_of_first_span:]
            consensus = ' '.join(re.split("\<.+?\>", consensus_element)).strip()
            consensus = ' '.join(consensus.split())  # Remove repeating whitespace within the consensus.

            return consensus

        def get_review_data(review_row):
            review_text = review_row.css("div.the_review::text").extract_first().strip()
            review_score_element_text = " ".join(review_row.css("div.review_desc div.small.subtle::text")
                                                           .extract())
            review_score = review_score_element_text.strip(" |OriginalScore:")
            return {"text": review_text, "rating:": review_score}

        if dvd_streaming_all_url[:-1] in response.url:
            # Handle browsing from list of movies to the page of each movie.
            print("Inside {}".format(response.url))
            json_response = json.loads(response.body_as_unicode())
            results = json_response['results']
            for href in [results[x]['url'] for x in range(0, len(results))]:
                print("Inside navigating to movie page {}.".format(href))
                yield response.follow(href, self.parse)

            # Handle getting next pages in the list of movies.
            if self.pageNum <= MAX_PAGE_NUM:
                old_page_num = self.pageNum
                self.pageNum += 1
                yield response.follow(dvd_streaming_all_url[:-1] + str(old_page_num), self.parse)

        # Gather general metadata about the current movie.
        if re.match("^.+\/m\/[^\/]+$", response.url):  # The page of a movie.
            print("Inside gathering movie metadata.")
            old_id = self.id
            self.id += 1
            movie_data = {
                "id": old_id,
                "name": extract_name(),
                "year": extract_with_css("span.h3.year::text").strip("()"),
                "summary": extract_summary(),
                "tomatometer": extract_with_css("span.meter-value.superPageFontColor span::text"),
                "average_rating": response.css("#scoreStats div.superPageFontColor::text").extract()[1].strip()[:-3]
            }

            # Handle navigating to first page of reviews.
            first_review_page_url = response.css('a.view_all_critic_reviews::attr(href)').extract_first()
            yield response.follow(first_review_page_url, callback=self.parse, meta=movie_data)

        # Handle gathering data from the current page of reviews.
        if re.match(r"https://www.rottentomatoes.com/m/.+?/reviews/.*", response.url):
            print("Inside gathering review data.")
            reviews = []
            for review_row in response.css("div.row.review_table_row"):
                reviews.append(get_review_data(review_row))
            if "reviews" not in response.meta.keys():
                response.meta["reviews"] = reviews
            else:
                response.meta["reviews"].extend(reviews)

            # Handle navigating to the next page of reviews (if it exists).
            pagination_info = response.css("span.pageInfo::text").extract_first().split()
            curr_page = int(pagination_info[1])
            last_page = int(pagination_info[3])
            if curr_page < last_page:
                yield response.follow(re.findall(r"/m/.+?/reviews/", response.url)[0] + "?page={}"
                                        .format(int(curr_page)+1),
                                      callback=self.parse,
                                      meta=response.meta)
            else:
                yield {
                        'id': response.meta['id'],
                        'name': response.meta['name'],
                        'year': response.meta['year'],
                        'summary': response.meta['summary'],
                        'tomatometer': response.meta['tomatometer'],
                        'average_rating': response.meta['average_rating'],
                        'reviews': response.meta['reviews']
                       }
