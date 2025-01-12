import sys
import json
import time
import pathlib

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.action_chains import ActionChains

import requests

DATA_FOLDER = './data'
DELAY = 0

def save_image(url, country, landmark):
    image_id = url.split('/')[-1]

    headers = {
        'User-Agent': 'unesco_whc_scraper (+https://antonrufino.github.io)'
    }

    image_data = requests.get(url, headers=headers).content
    with open(f'{DATA_FOLDER}/{country}/{landmark}/{image_id}.jpg', 'wb') as imgf:
        imgf.write(image_data)


def check_copyright(driver):
    image_properties = [prop.text for prop in driver.find_elements(By.CSS_SELECTOR, 'div.py-4 > div.text-white')]
    for prop in image_properties:
        if prop in ['Copyright: © UNESCO', 'Copyright: Public Domain']:
            print(prop)
            return True
    
    return False


def get_image_data(driver, gallery_url, country, landmark, cc_only=True):
    image_data_list = []

    print(f'Current URL: {gallery_url}')
    driver.get(gallery_url)
    try:
        see_all_link = driver.find_element(By.CSS_SELECTOR, 'ul.pagination > li:last-child > a')
        see_all_link.click()
    except NoSuchElementException:
        print("Only one page...")

    pic_links = driver.find_elements(By.CSS_SELECTOR, 'div.icaption-gallery a')
    pic_urls = [link.get_attribute('href') for link in pic_links]

    for url in pic_urls:
        print(f'Modal url: {url}')
        driver.get(url)

        # Copyright check
        if not cc_only or check_copyright(driver):
            image_info_url = driver.find_element(By.CSS_SELECTOR, 'div.py-4 > div.text-white a').get_attribute('href')
            image_url = f'https://whc.unesco.org/document/{image_info_url.split('/')[-1]}'
            print(f'Image link: {image_info_url}')

            image_data_list.append({
                'source_url': image_info_url,
                'description': driver.find_element(By.CSS_SELECTOR, 'div.py-4 > div:nth-child(1)').text
            })
            save_image(image_url, country, landmark)
        else:
            print('Copyright not allowable')

    return image_data_list


def main():
    with open(sys.argv[1]) as f:
        data = [json.loads(line) for line in f]

    chrome_options = Options()
    chrome_options.add_argument("disable-quic")
    chrome_options.add_argument('--user-agent=unesco_whc_scraper (+https://antonrufino.github.io)')

    driver = webdriver.Chrome(options=chrome_options)
    driver.set_page_load_timeout(30)
    driver.maximize_window()

    try:
        with open('sea-landmarks-images.jsonl', 'w') as f:
            for d in data:
                for landmark in d['landmarks']:
                    pathlib.Path(f"{DATA_FOLDER}/{d['country']}/{landmark['name']}").mkdir(parents=True, exist_ok=True)
                    landmark['images'] = get_image_data(
                        driver,
                        landmark['gallery_link'],
                        d['country'],
                        landmark['name'],
                        cc_only=False
                    )
                
                f.write(json.dumps(d) + '\n')
    except Exception:
        driver.save_screenshot('./error.png')
    finally:
        driver.close()


if __name__ == '__main__':
    main()