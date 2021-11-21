import csv
import requests
from bs4 import BeautifulSoup

if __name__ == "__main__": 
    # Specify the corresponding URL
    id = 0
    with open('./clean_data.csv', 'w', encoding='utf-8') as fp:
        writer = csv.writer(fp)
        header = ['id', 'price', 'address', 'bed','bath','type']
        writer.writerow(header)
        for i in range(43):
            url = 'https://www.daft.ie/property-for-rent/dublin-city?pageSize=20&page=' + str(i+1)
            headers = {
                'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.69 Safari/537.36 '
            }
            # Initiate a request
            # The method of get will return the response object
            response = requests.get(url=url, headers=headers)
            # Get the response data, text will return the string form of response data
            page_text = response.text
            soup = BeautifulSoup(page_text, 'lxml')
            # Extracting the elements of one web page
            item = soup.find_all(attrs={'class': 'SearchPage__Result-gg133s-2 itNYNv'})
            for x in item:
                soup2 = BeautifulSoup(str(x),'lxml')
                address = soup2.find_all(attrs={'class': 'TitleBlock__Address-sc-1avkvav-7 eARcqq'})
                price_all = soup2.find_all(attrs={'class': 'SubUnit__Title-sc-10x486s-5 keXaVZ'})
                bed_bath_type_all = soup2.find_all(attrs={'class': 'SubUnit__CardInfoItem-sc-10x486s-7 hHMkFR'})

                #format address
                address = str(address[0]) 
                tmp = address.split(', ')
                address = tmp[len(tmp)-1]
                tmp = address.split(' ')
                address = tmp[1].replace('</p>','')
                if 'Dublin' in address:
                    #use number 19 for Co. Dublin because other area has number for district, e.g. Dublin 2
                    address = '19'   
                elif '6W' in address:
                    #use number 6.5 to represent Dublin 6W
                    address = '6.5'
                # Save the data into a file
                for num in range(len(price_all)):
                    price = str(price_all[num]).replace('<p class="SubUnit__Title-sc-10x486s-5 keXaVZ" data-testid="sub-title">',
                                                            '').replace(
                            '</p>', '').replace('€','').replace(',','')
                    if 'per month' in price:
                        price = price.replace(' per month', '')
                    else:
                        tmp = price.replace(' per week', '')
                        price = round(tmp * 52 / 12) #convert to monthly rent, 1 year = 52 weeks = 12 months

                    bed_bath_type = str(bed_bath_type_all[num*2]).replace('<div class="SubUnit__CardInfoItem-sc-10x486s-7 hHMkFR">',
                                                '').replace('<!-- -->','').replace('</div>','')
                    if ' · ' in bed_bath_type:
                        bed = bed_bath_type.split(' · ')[0].replace(' Bed', '')
                        bath = bed_bath_type.split(' · ')[1].replace(' Bath','')
                        type = bed_bath_type.split(' · ')[2]
                        if type == 'Apartment':
                            type = '1'
                        else:
                            type = '2'
                    else:
                        bed = '1'
                        bath = '1'
                        type = bed_bath_type
                    id += 1
                    row = [id, price, address, bed, bath, type]
                    writer.writerow(row)
                    #print(row)

            
        print('Success!' + str(i))
