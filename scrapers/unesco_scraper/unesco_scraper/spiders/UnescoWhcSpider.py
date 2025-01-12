import scrapy

sea_vqa_countries = [
    'cambodia',
    'indonesia',
    'lao',
    'malaysia',
    'philippines',
    'singapore',
    'thailand',
    'viet'
]

class UnescoWhcSpider(scrapy.Spider):
    name = 'unesco_whc'
    start_urls = [
        'https://whc.unesco.org/en/list/'
    ]

    def parse(self, response):
        country_names = response.css('#acc > .card-body > h4 > a::text').getall()
        country_lists = response.css('#acc > .card-body > div.list_site > ul')

        country_data = list(zip(country_names, country_lists))

        for country, landmark_list in country_data:
            if country.lower().split(' ')[0] not in sea_vqa_countries: continue
            
            landmarks = []
            for landmark_link in landmark_list.css('li > a:first-child'):
                landmark = {
                    "name": ''.join(landmark_link.css('::text').getall()),
                    "gallery_link": response.urljoin(landmark_link.css('::attr(href)').get() + '/gallery')
                }
                landmarks.append(landmark)
            
            yield {
                'country': country,
                'landmarks': landmarks
            }
        