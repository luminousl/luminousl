#!/usr/bin/env python
# coding: utf-8

import markdown2
import os, sys
import argparse
# import markdown


def markdown2html(md_path, html_path, css_path='markdown_stlye/github_style_md2.css'):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    abs_css_path = os.path.join(current_dir, css_path)
    output = """<!DOCTYPE html>
    <html lang="en">

    <head>
        <meta charset="utf-8">
        <style type="text/css">
    """

    current_directory = os.getcwd()
    with open(abs_css_path, 'r') as cssin:
        output += cssin.read()

    output += """
        </style>
    </head>

    <body>
    """

    with open(md_path, 'r') as mdin:
        output += markdown2.markdown(mdin.read(), extras=['tables'])

    output += """</body>
    </html>
    """

    with open(html_path,'w') as outfile:
        outfile.write(output)


def main():
    parser = argparse.ArgumentParser(description="该工具将markdown文件转换为html文件，支持css样式")
    parser.add_argument("--markdown", "-m", required=True, help="markdown文件路径")
    parser.add_argument("--html", "-t", required=True, help="html文件路径")
    parser.add_argument("--css", "-c", default='', help="css文件路径")
    args = parser.parse_args()

    if args.css:
        markdown2html(args.markdown, args.html, args.css)
    else:
        markdown2html(args.markdown, args.html)


if __name__=='__main__':
    main()
    