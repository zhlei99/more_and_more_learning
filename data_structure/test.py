#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 16:17:51 2019

@author: zhaolei
"""
def get_url_data(url):
    req = request.Request(url)
    req.add_header('User-Agent','Mozilla/6.0(iPhone; CPU iPhone OS 8_0 like Mac OS X) AppleWebKit/536.26 (KHTML, like Gecko) Version/8.0 Mobile/10A5376e Safari/8536.25')
    with request.urlopen(req) as f:
        print("Status:",f.status,f.reason)
        for k,v in f.getheaders():
            print('%s:%s'%(k,v))
        return f.read().decode('utf-8')

class MyHTMLParser_example(HTMLParser):
    
    def __init__(self):
        super().__init__()
        self._parsedata = '' #设置一个属性标志位。
        self.info = []        
    
    def handle_starttag(self, tag, attrs):
        
        if ('class', 'event-title') in attrs:
            self._parsedata = 'name'
        if ('class', 'say-no-more') in attrs:
            self._parsedata = 'year'
        if 'time' == tag:
            self._parsedata = 'time' 
        if ('class', 'event-location') in attrs:
            self._parsedata = 'loction'
            
    def handle_endtag(self, tag):
        self._parsedata = ''#在HTML 标签结束时，把标志位清空
    
    def handle_data(self, data):
        
        if self._parsedata == 'name':
            #通过标志位判断输出名字
            self.info.append(f'会议名称：{data}')
        
        if self._parsedata == 'year':
            if re.match(r'\s\d{4}', data): # 因为后面还有两组 say-no-more 后面的data却不是年份信息,所以用正则检测一下
                self.info.append(f'会议年份：{data}')
            
        if self._parsedata == 'time':
            self.info.append(f'会议时间：{data}')
        
def main_html_example():
    url = 'https://www.python.org/events/python-events/'
    html_data = get_url_data(url)
    parser = MyHTMLParser_example()
    parser.feed(html_data)
    for s in parser.info:
        print(s)