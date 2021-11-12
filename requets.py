# -*- coding = utf-8 -*-
# @Time : 2021/11/9 11:32 下午
# @Author : Kiser
# @File : requets.py
# @Software : PyCharm
import csv
import requests
from bs4 import BeautifulSoup

if __name__ == "__main__":
    # Specify the corresponding URL
    number = 1
    with open('./data.csv', 'w', encoding='utf-8') as fp:
        writer = csv.writer(fp)
        header = ['id', 'price', 'address', 'bedroom', 'bathroom', 'area']
        writer.writerow(header)
        for i in range(136):
            url = 'https://www.daft.ie/property-for-sale/dublin-city?pageSize=20&from=' + str(i * 20)
            headers = {
                'User-Agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) '
                              'Chrome/95.0.4638.69 Mobile Safari/537.36'
            }
            # Initiate a request
            # The method of get will return the response object
            response = requests.get(url=url, headers=headers)
            # Get the response data, text will return the string form of response data
            page_text = response.text
            soup = BeautifulSoup(page_text, 'lxml')
            # Extracting the elements of one web page
            price_all = soup.find_all(attrs={'class': 'TitleBlock__StyledSpan-sc-1avkvav-4 gDBFnc'})
            address_all = soup.find_all(attrs={'class': 'TitleBlock__Address-sc-1avkvav-7 knPImU'})
            bedroom_all = soup.find_all(attrs={'class': 'TitleBlock__CardInfoItem-sc-1avkvav-8 jBZmlN'})
            bathroom_all = soup.find_all(attrs={'data-testid': 'baths'})
            area_all = soup.find_all(attrs={'data-testid': 'floor-area'})

            # Save the data into a file
            for num in range(len(price_all)):
                id = str(number)
                if len(price_all) >= num + 1:
                    price = str(price_all[num]).replace('<span class="TitleBlock__StyledSpan-sc-1avkvav-4 gDBFnc">',
                                                        '').replace(
                        '<!-- --> </span>', '')
                else:
                    price = ''
                if len(address_all) >= num + 1:
                    address = address_all[num].string
                else:
                    address = ''
                if len(bedroom_all) >= num + 1:
                    bedroom = bedroom_all[num].string
                else:
                    bedroom = ''
                if len(bathroom_all) >= num + 1:
                    bathroom = bathroom_all[num].string
                else:
                    bathroom = ''
                if len(area_all) >= num + 1:
                    area = area_all[num].string
                else:
                    area = ''
                number += 1
                row = [id, price, address, bedroom, bathroom, area]
                writer.writerow(row)
                print(row)
        print('Success!' + str(i))
