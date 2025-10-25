import sys
import pymupdf
import os
import re
import unicodedata
from dataclasses import dataclass
from collections import Counter
import logging
import pandas as pd

os.environ["TESSDATA_PREFIX"] = "./training"


class Cleaner:

    @staticmethod
    def clean_page_numbers(lines: list) -> list:

        logging.debug(f"Cleaning page numbers")

        try:
            PAGE_NUMBER_PATTERN = re.compile(r'\s*\d+\s*')

            return [
                line for line in lines if not PAGE_NUMBER_PATTERN.fullmatch(line.text)
            ]

        except Exception as e:
            logging.exception(f"Error cleaning page numbers: {e}")
            raise

    @staticmethod
    def join_broken_sentences(lines: list) -> str:

        logging.debug(f"Joining broken sentences.")

        try:
            BROKEN_WORD_PATTERN = re.compile(r'\w-\s*$')
            HYPHEN_END_PATTERN = re.compile(r'-\s*$')

            page_line_text = []
            skip_until = -1  # highest index of merged lines

            for i, current_line in enumerate(lines):
                # Skip lines already merged into previous text
                if i <= skip_until:
                    continue

                current_text = current_line.text

                # If current line ends with a broken word, merge forward
                if BROKEN_WORD_PATTERN.search(current_text):
                    j = i + 1
                    # Keep merging as long as there are more broken lines
                    while j < len(lines):
                        current_text = HYPHEN_END_PATTERN.sub('', current_text) + lines[j].text.lstrip()
                        if BROKEN_WORD_PATTERN.search(current_text) and j + 1 < len(lines):
                            j += 1
                            continue
                        else:
                            break

                    skip_until = j  # mark lines up to j as consumed
                    page_line_text.append(current_text)
                else:
                    page_line_text.append(current_text)

            return " ".join(page_line_text)

        except Exception as e:
            logging.exception(f"Error joining broken sentences: {e}")
            raise

    @staticmethod
    def clean_extracted_text(text: str, ocr: bool, multipage_parentheses: str | None = None) -> tuple[str, str]:
        """Clean text in a single pass for better performance."""

        try:
            result = []
            i = 0

            while i < len(text):
                char = text[i]

                # Skip parentheses and their content
                pairs = {'(': ')', '[': ']', '{': '}'}

                if char in pairs or multipage_parentheses is not None:

                    if multipage_parentheses:
                        open_char = multipage_parentheses
                    else:
                        open_char = char

                    close_char = pairs[open_char]
                    depth = 1
                    i += 1
                    iterations = 0
                    while i < len(text) and depth >= 1:

                        if iterations >= 30 and ocr is True: # Misrecognized parentheses
                            i -= 30
                            result.append(open_char)
                            break

                        if text[i] == open_char:
                            depth += 1
                        elif text[i] == close_char:
                            depth -= 1
                            multipage_parentheses = None
                        elif i == len(text) - 1:
                            multipage_parentheses = open_char

                        i += 1
                        iterations += 1

                    if result and result[-1] == ' ': # Remove extra space
                        result.pop()

                    continue

                # Skip emojis and symbols
                if (unicodedata.category(char)[0] in ['S'] or
                        (0x1F300 <= ord(char) <= 0x1FAFF)):
                    i += 1
                    continue

                # Normalize Unicode and add to result
                normalized = unicodedata.normalize("NFKD", char)
                result.append(normalized)
                i += 1

            return ''.join(result), multipage_parentheses

        except Exception as e:
            logging.exception(f"Error cleaning sentences: {e}")
            raise

@dataclass
class StyledLine:
    text: str
    font_size: float
    font_name: str
    start_x: float
    start_y: float
    end_x: float

    def __post_init__(self):
        self.font_size = round(self.font_size)
        self.start_x = round(self.start_x)
        self.start_y = round(self.start_y)
        self.end_x = round(self.end_x)

class TextHeuristics:
    def __init__(self) -> None:
        self.threshold = None

    @staticmethod
    def get_styling_counter(lines: list, attribute: str) -> Counter:

        try:
            counter = Counter()
            for line in lines:
                attr_value = getattr(line, attribute)
                counter[attr_value] += len(line.text)
            return counter

        except AttributeError as e:
            logging.exception(f"Invalid styling attribute: {attribute}")
            raise
        except Exception as e:
            logging.exception(f"Error getting style counter for {attribute}: {e}")
            raise

    @staticmethod
    def most_common_value(counter: Counter):

        return counter.most_common(1)[0][0] if counter else None

    def compute_bounds(self, data) -> tuple[float, float]:
        series = pd.Series(data)
        mean = series.mean()
        std = series.std()

        if std == 0 or pd.isna(std):
            return series.min(), series.max()

        z_scores = (series - mean) / std
        inlier_series = series[z_scores.abs() <= self.threshold]

        if inlier_series.empty:
            return series.min(), series.max()

        return inlier_series.min(), inlier_series.max()

    @staticmethod
    def compute_differences(data):

        return [abs(data[i] - data[i+1]) for i in range(len(data)-1)]

    def compute_word_gaps(self, lines):

        start_x_differences = []

        for _, group in pd.DataFrame(lines).groupby("start_y"):
            start_x_differences.extend(TextHeuristics.compute_differences(group["start_x"].tolist()))

        return self.compute_bounds(data=sorted(start_x_differences))

    def compute_line_gaps(self, start_y_counter):

        differences = TextHeuristics.compute_differences(sorted(start_y_counter))

        return self.compute_bounds(data=differences)

    def set_threshold(self, ocr):

        if ocr:
            self.threshold = 3.0
        else:
            self.threshold = 1.0

    def analyze(self, lines: list, ocr: bool) -> dict:

        counters = {
            attr: self.get_styling_counter(lines, attr)
            for attr in ['font_size', 'font_name', 'start_x', 'start_y']
        }

        most_common = {k: self.most_common_value(v) for k, v in counters.items()}

        self.set_threshold(ocr=ocr)

        font_sizes = [size for size, freq in counters['font_size'].items() for _ in range(freq)]
        font_bounds = self.compute_bounds(font_sizes)

        if ocr:
            word_gaps = self.compute_word_gaps(lines=lines)
        else:
            word_gaps = [None, None]
        line_gaps = self.compute_line_gaps(counters['start_y'])


        return {'start x': {'most common': most_common['start_x'], 'minimum': min(counters['start_x']), 'maximum': max(counters['start_x']), 'lower bound': word_gaps[0], 'upper bound': word_gaps[1]},
                'start y': {'most common': most_common['start_y'], 'minimum': min(counters['start_y']), 'maximum': max(counters['start_y']), 'lower bound': line_gaps[0], 'upper bound': line_gaps[1]},
                'end x': {'most common': most_common['end_x'], 'minimum': min(counters['end_x']), 'maximum': max(counters['end_x']), 'lower bound': word_gaps[0], 'upper bound': word_gaps[1]},
                'font size': {'most common': most_common['font_size'], 'lower bound': font_bounds[0], 'upper bound': font_bounds[1]},
                'font name': {'most common': most_common['font_name']}}


class DocumentAnalysis:

    def __init__(self):
        self.page_heuristics = None
        self.bottom_boundary = None
        self.top_boundary = None
        self.left_boundary = None

    @staticmethod
    def get_page_blocks_from_dict(pdf: pymupdf.Document, page_number: int, sort: bool) -> list:

        try:
            page_text = pdf[page_number].get_textpage()
            page_dict = page_text.extractDICT(sort=sort)
            page_blocks = page_dict["blocks"]

            return page_blocks

        except Exception as e:
            logging.exception(f"Error reading PDF blocks: {e}")
            raise

    @staticmethod
    def iter_pdf_styling_from_blocks(page_blocks: list) -> Generator[StyledLine]:

        try:
            for block in page_blocks:
                if block["type"] != 0:
                    continue # text blocks only

                for line in block["lines"]:
                    for span in line["spans"]:
                        text = "".join(span["text"]).strip()
                        if text:
                            yield StyledLine(text, span["size"], span["font"], span["origin"][0], span["origin"][1], span["bbox"][2])


        except Exception as e:
            logging.exception(f"Error reading styles from PDF blocks: {e}")
            raise

    # @staticmethod
    # def separate_by_paragraph(lines_with_styling: list):

    @staticmethod
    def check_ocr(lines: list[StyledLine]) -> bool:

        if len(lines) == 0:
            return False

        words = 1
        phrases = 1

        for i, line in enumerate(lines):

            if line.text.strip() is None:
                continue
            elif " " not in line.text.strip():
                words += 1
            else:
                phrases += 1

        if words / phrases > 0.75:
            return True
        else:
            return False

    def is_at_left_margin(self, line: StyledLine) -> bool:
        return line.start_x == self.left_boundary

    def is_after_left_margin(self, line: StyledLine) -> bool:
        return line.start_x > self.left_boundary

    def is_before_left_margin(self, line: StyledLine) -> bool:
        return line.start_x < self.left_boundary

    def is_footer_region(self, line: StyledLine) -> bool:
        return line.start_y >= self.bottom_boundary - self.page_heuristics['start y']['lower bound']

    def is_header_region(self) -> bool:
        return self.top_boundary is None

    def is_dominant_word_gap(self, current_word: StyledLine, next_word: StyledLine) -> bool:
        word_separation = next_word.start_x - current_word.end_x
        return self.page_heuristics['start x']['lower bound'] <= word_separation <= self.page_heuristics['start x']['upper bound']

    def is_dominant_font(self, line: StyledLine) -> bool:
        return self.page_heuristics['font size']['lower bound'] <= line.font_size <= self.page_heuristics['font size']['upper bound']

    def set_page_boundaries(self) -> None:

        self.left_boundary = self.page_heuristics['start x']['most common']
        self.bottom_boundary = self.page_heuristics['start y']['maximum']

    def setup(self, lines: list[StyledLine], ocr: bool) -> None:

        page_heuristics = TextHeuristics()

        self.page_heuristics = page_heuristics.analyze(lines=lines, ocr=ocr)

        if ocr and self.page_heuristics['font name']['most common'] != 'GlyphLessFont':
            ocr = False
            self.page_heuristics = page_heuristics.analyze(lines=lines, ocr=ocr)

        self.set_page_boundaries()

    def filter_by_boundaries(self, lines, ocr):

        def skip_line(i: int, y_boundary: float) -> int:
            while i < len(lines) and lines[i].start_y == y_boundary:
                i += 1
            return i

        def collect_line(i: int, y_boundary: float) -> tuple[list[StyledLine], int]:
            current_line = []
            while i <= len(lines) - 1 and lines[i].start_y == y_boundary:
                current_line.append(lines[i])
                i += 1
            return current_line, i

        def collect_once(i: int) -> int:
            current_line.append(lines[i])
            return i + 1


        filtered_lines: list[StyledLine] = []
        current_line: list[StyledLine] = []

        self.setup(lines, ocr)

        i = 0
        while i < len(lines):

            current_word: StyledLine = lines[i]
            line_y_boundary = current_word.start_y

            if self.is_header_region():

                if self.is_before_left_margin(line=current_word):  # Header
                    i = skip_line(i, line_y_boundary)

                if self.is_at_left_margin(line=current_word):  # Body start

                    gap_to_next_line = 0
                    j = i
                    while gap_to_next_line == 0:
                        gap_to_next_line = lines[j + 1].start_y - lines[i].start_y
                        j += 1

                    if gap_to_next_line > self.page_heuristics['start y']['upper bound']:  # Aligned header
                        i = skip_line(i, line_y_boundary)

                    else:
                        self.top_boundary = current_word.start_y
                        current_line, i = collect_line(i, line_y_boundary)

                elif self.is_after_left_margin(line=current_word):  # Edge case: Indented main body

                    if ocr and current_word.start_x - self.page_heuristics['start x']['lower bound'] > self.left_boundary:  # Indented header
                        i = skip_line(i, line_y_boundary)

                    elif self.is_dominant_font(line=current_word):

                        self.top_boundary = current_word.start_y
                        current_line, i = collect_line(i, line_y_boundary)

                    else:
                        i = skip_line(i, line_y_boundary)

                else:
                    i = skip_line(i, line_y_boundary)

            elif self.is_footer_region(line=current_word):  # Very bottom

                if current_word.start_x == self.page_heuristics['start x']["most common"]:
                    current_line, i = collect_line(i, line_y_boundary)

                elif lines[i].start_x == filtered_lines[-1].start_x:  # Continued indented block

                    while lines[i].start_y <= line_y_boundary:

                        if i == len(lines) - 1:
                            i = collect_once(i)
                            break
                        else:

                            if ocr and self.is_dominant_word_gap(current_word=lines[i], next_word=lines[i + 1]): # Doesn't work for non-ocr
                                i = collect_once(i)
                            else:  # Replace with table detection
                                i = collect_once(i)
                                break

                else:
                    i = skip_line(i, line_y_boundary) # Footer

            elif self.is_at_left_margin(line=current_word):  # Main body

                if self.is_dominant_font(line=current_word):
                    current_line, i = collect_line(i, line_y_boundary)

                else:  # Edge case: Aligned title
                    i = skip_line(i, line_y_boundary)

            elif lines[i].start_y < filtered_lines[-1].start_y:  # Titles outside regular read-order

                i = skip_line(i, line_y_boundary)

            elif self.is_after_left_margin(current_word):  # Indented block

                if i == len(lines) - 1 and lines[i].start_y == line_y_boundary:
                    i = collect_once(i)

                elif i < len(lines) - 1:
                    while lines[i].start_y == line_y_boundary:

                        if ocr and self.is_dominant_word_gap(current_word=lines[i], next_word=lines[i + 1]):
                            current_line, i = collect_line(i, line_y_boundary)
                        else:
                            i = collect_once(i)
                            break

            elif self.is_before_left_margin(line=current_word):  # Left-side footer
                i = skip_line(i, line_y_boundary)

            else:
                i += 1


            if len(current_line) > 0:
                # for word in current_line:

                filtered_lines.append(StyledLine(text=' '.join(line.text for line in current_line if line.text.strip()),
                                                 font_size=pd.Series([line.font_size for line in current_line]).mean(),
                                                 font_name=current_word.font_name,
                                                 start_x=current_word.start_x,
                                                 start_y=current_word.start_y,
                                                 end_x=current_word.end_x))
                current_line = []

        return filtered_lines


class PDFReader:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.pdf = None


    def __enter__(self) :
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.pdf:
            self.close()

        if exc_type:
            logging.error(f"An exception occurred: {exc_val}")

        return False

    def iter_pages(self, sort=False):
        for i in range(self.get_page_count()):
            yield DocumentAnalysis.get_page_blocks_from_dict(pdf=self.pdf, page_number=i, sort=sort)

    def get_page_count(self) -> int:
        return self.pdf.page_count

    def open(self):
        if self.pdf is None:
            self.pdf = pymupdf.open(str(self.pdf_path))
            logging.debug(f"Opened PDF: {self.pdf_path}")
            return self.pdf
        return None

    def close(self):
        if self.pdf:
            try:
                self.pdf.close()
                logging.debug(f"Closed PDF: {self.pdf_path}")
            except Exception as e:
                logging.error(f"Error closing PDF: {e}")
            finally:
                self.pdf = None


class OutputWriter:
    def __init__(self):
        self.output_path = None

    def set_output_path(self, pdf: pymupdf.Document, pdf_path: str) -> str:

        if len(pdf.metadata['title']) > 1:
            sanitized_title = re.sub(r'[<>:"/\\|?*\n\r\t;]', '_', pdf.metadata['title']).strip()
            self.output_path = f"./generated/{sanitized_title}.txt"

        else:
            base_name = os.path.splitext(os.path.basename(pdf_path))[0]
            self.output_path = f"./generated/{base_name}.txt"

        return self.output_path

    def write(self, mode: str, text: str | None = None):
        with open(self.output_path, mode, encoding='utf-8') as f:
            if text is not None:
                f.write(text)


def main():

    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        pdf_path = "./docs/test_OCR.pdf"

    with PDFReader(pdf_path) as pdf_reader:

        output_writer = OutputWriter()
        output_writer.set_output_path(pdf=pdf_reader.pdf, pdf_path=pdf_path)

        output_writer.write(mode="w")

        multipage_parentheses = None
        for page_blocks in pdf_reader.iter_pages(sort=False):


            lines_with_styling = list(DocumentAnalysis.iter_pdf_styling_from_blocks(page_blocks=page_blocks))
            ocr = False
            if DocumentAnalysis.check_ocr(lines_with_styling=lines_with_styling):
                ocr = True

            lines_with_styling = DocumentAnalysis.filter_by_boundaries(lines_with_styling=lines_with_styling, ocr=ocr)
            lines_without_numbers = Cleaner.clean_page_numbers(lines=lines_with_styling)
            cleaned_text = Cleaner.join_broken_sentences(lines=lines_without_numbers)
            page_text, multipage_parentheses = Cleaner.clean_extracted_text(text=cleaned_text,
                                                                                 multipage_parentheses=multipage_parentheses, ocr=ocr)

            output_writer.write(mode="a", text=f'{page_text}\n\n')

    pdf_reader.close()

if __name__ == '__main__':
    main()