"""
Tests for the documentation: the contents list at the top of README.md has every section, in the order of the file, and
every link (and image) of README.md and of the pages of docs/ reaches an existing file and, when it names one, an existing
heading (GitHub makes the anchor of a heading from its text)

Run from the project folder with:  python -m unittest tests.test_readme -v
"""

import re
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
README = PROJECT / 'README.md'
DOCS = PROJECT / 'docs'
FENCE = re.compile(r'^```[A-Za-z]*\s*$')


def github_anchor(title):
    """The anchor that GitHub gives to a heading: lower case, no punctuation, spaces as hyphens"""
    return re.sub(r'[^\w\- ]', '', title.lower()).replace(' ', '-')


def pages():
    """The files of the documentation: README.md and the pages of docs/"""
    return [README] + sorted(DOCS.glob('*.md'))


def read_page(path):
    """
    (headings, links): the headings of the page outside the code blocks as (level, title, anchor), and what it points to
    as (text, target): the markdown links and images, and the src and srcset of the html
    """
    headings, links, seen, in_code = [], [], {}, False
    for line in path.read_text(encoding='utf-8').splitlines():
        if FENCE.match(line):
            in_code = not in_code
            continue
        if in_code:
            continue
        heading = re.match(r'^(#{1,6}) (.+?)\s*$', line)
        if heading:
            anchor = github_anchor(heading.group(2))
            count = seen.get(anchor, 0)
            seen[anchor] = count + 1
            headings.append((len(heading.group(1)), heading.group(2), anchor if not count else f'{anchor}-{count}'))
        text = re.sub(r'`+[^`]*`+', '', line)                              # what is written as code is not a link
        links += re.findall(r'\[([^\]]*)\]\(([^)\s]+)\)', text)
        links += [('html', target) for target in re.findall(r'(?:src|srcset)="([^"]+)"', text)]
    return headings, links


def contents_of_readme():
    """The links to anchors of the contents list of README.md as (level, text, anchor), the level from the indentation"""
    contents, in_code = [], False
    for line in README.read_text(encoding='utf-8').splitlines():
        if FENCE.match(line):
            in_code = not in_code
        link = None if in_code else re.match(r'^( *)- \[(.+?)\]\(#([^)]+)\)\s*$', line)
        if link:
            contents.append((2 + len(link.group(1)) // 2, link.group(2), link.group(3)))
    return contents


class ContentsTests(unittest.TestCase):

    def test_the_contents_list_is_there(self):
        self.assertTrue(contents_of_readme(), 'the contents list at the top of README.md is missing')

    def test_every_section_is_listed_in_the_order_of_the_file(self):
        headings, _ = read_page(README)
        sections = [(level, title, anchor) for level, title, anchor in headings if level in (2, 3)]
        self.assertEqual(sections, contents_of_readme(),
                         'the contents list must have the headings of level 2 and 3 of README.md, in order (a new section, '
                         'a renamed one or a different order needs the list at the top of README.md changed)')


class LinkTests(unittest.TestCase):

    def test_there_are_pages(self):
        self.assertGreater(len(pages()), 1, 'the pages of docs/ are missing')

    def test_every_link_reaches_a_file_and_a_heading(self):
        anchors = {page: {anchor for _, _, anchor in read_page(page)[0]} for page in pages()}
        for page in pages():
            for text, target in read_page(page)[1]:
                if target.startswith(('http://', 'https://', 'mailto:')):
                    continue
                path, _, anchor = target.partition('#')
                destination = (page.parent / path).resolve() if path else page
                with self.subTest(page=page.name, link=target):
                    self.assertTrue(destination.exists(), f'{page.name}: "{text}" points to {target}, which is not there')
                    if anchor and destination.suffix == '.md':
                        known = anchors.get(destination) or {a for _, _, a in read_page(destination)[0]}
                        self.assertIn(anchor, known, f'{page.name}: "{text}" points to a heading that is not in '
                                                     f'{destination.name}: #{anchor}')


class WebsiteTests(unittest.TestCase):
    """The website (_config.yml, GitHub Pages): every page of docs/ is in the side bar, in the order of the table of the README"""

    def test_every_page_is_in_the_side_bar_in_the_order_of_the_readme(self):
        config = (PROJECT / '_config.yml').read_text(encoding='utf-8')
        pages_in_bar = re.findall(r'scope: \{path: "([^"]+)"\}\s*\n\s*values: \{title: "[^"]+", nav_order: (\d+)', config)
        order = [path for path, _ in sorted(pages_in_bar, key=lambda item: int(item[1]))]
        self.assertEqual(order[0], 'README.md')
        self.assertEqual(sorted(order[1:]), sorted(f'docs/{page.name}' for page in DOCS.glob('*.md')),
                         'every page of docs/ needs its line in the defaults of _config.yml')
        table = re.findall(r'^\| \[[^\]]+\]\((docs/[^)]+\.md)\) \|', README.read_text(encoding='utf-8'), flags=re.M)
        self.assertEqual(order[1:], table, 'the side bar must follow the table of the documentation in README.md')
        self.assertEqual(len({number for _, number in pages_in_bar}), len(pages_in_bar), 'two pages have one place')


if __name__ == '__main__':
    unittest.main()
