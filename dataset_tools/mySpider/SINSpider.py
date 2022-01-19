import os
import sys
import xlrd
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
# import copy
from xlutils.copy import copy
import time
from lxml import etree
# from xml import etree
# import xml.etree.ElementTree as ET


def get_excel_data(sheetName, row, col=0):
    excelDir = './供应商.xlsx'
    workBook = xlrd.open_workbook(excelDir)
    workSheet = workBook.sheet_by_name(sheetName)

    return workSheet.cell(row, col).value


def set_excel_data():
    excelDir = './供应商.xlsx'
    workBook = xlrd.open_workbook(excelDir)
    newWorkBook = copy(workBook)
    newSheet = newWorkBook.get_sheet(0)

    return newWorkBook, newSheet


def spider(data):
    url = 'https://xin.baidu.com'
    driver.get(url)

    driver.find_element_by_id("aqc-search-input").send_keys(data)
    driver.find_element_by_class_name("search-btn").click()
    time.sleep(3)

    html = etree.HTML(driver.page_source)
    name_cache = html.xpath(".//div[@class='wrap']/div[1]//h3/a/@title")
    print(name_cache)

    if name_cache:
        name_cache = name_cache[0]
    if name_cache != data:
        return

    p = html.xpath(".//div[@class='wrap']/div[1]//h3/a/@href")[0]
    new_url = f'https://xin.baidu.com{p}'
    driver.get(new_url)
    new_html = etree.HTML(driver.page_source)
    name = new_html.xpath(".//h2/text()")[0]
    code = new_html.xpath(".//table[@class='zx-detail-basic-table']/tbody/tr[4]/td[2]/text()")[0]

    return name, code


if __name__ == "__main__":
    driver = webdriver.Chrome('./chromedriver')

    workBook, sheet = set_excel_data()

    for row in range(27, 40):
        data = get_excel_data('input', row)

        name, code = (spider(data) if spider(data) else ('null', 'null'))

        if code != 'null':
            print('----- Succeed. -----')
            sheet.write(row, 1, name)
            sheet.write(row, 2, code)
        else:
            print('------ Null. ------')
            sheet.write(row, 1, 'null')
            sheet.write(row, 2, 'null')

    driver.quit()

    workBook.save('./供应商查询结果.xlsx')


