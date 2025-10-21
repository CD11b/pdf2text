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
    origin_x: int
    origin_y: int

class DocumentAnalysis:

    def __init__(self):
        self.font_heuristics = None

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
    def get_pdf_styling_from_blocks(page_blocks: list) -> list:

        try:
            lines_with_styling: list[StyledLine] = []

            for block in page_blocks:
                if block["type"] != 0:
                    continue # text blocks only

                for line in block["lines"]:
                    spans = line["spans"]

                    for span in spans:

                        line_text = "".join(span["text"]).strip()
                        font_size = span["size"]
                        font_name = span["font"]
                        origin_x = span["origin"][0]
                        origin_y = span["origin"][1]

                        lines_with_styling.append(StyledLine(text=line_text, font_size=font_size, font_name=font_name, origin_x=origin_x, origin_y=origin_y))

            return lines_with_styling

        except Exception as e:
            logging.exception(f"Error reading styles from PDF blocks: {e}")
            raise

    @staticmethod
    def get_styling_counter(lines_with_styling: list, styling_attribute: str) -> Counter:

        try:
            counter = Counter()
            for line in lines_with_styling:
                attr_value = getattr(line, styling_attribute)
                if isinstance(attr_value, float):
                    attr_value = round(attr_value)
                counter[attr_value] += len(line.text)
            return counter

        except Exception as e:
            logging.exception(f"Error getting style counter for {styling_attribute}: {e}")
            raise

    @staticmethod
    def get_n_most_common(counter: Counter, n: int) -> list:

        try:
            return counter.most_common(n)
        except Exception as e:
            logging.exception(f"Error finding most common style: {e}")
            raise

    @staticmethod
    def get_styling_bounds(data, threshold):
        series = pd.Series(data)
        mean = series.mean()
        std = series.std()

        if std == 0 or pd.isna(std):
            return series.min(), series.max()

        z_scores = (series - mean) / std
        inlier_series = series[z_scores.abs() <= threshold]

        if inlier_series.empty:
            return series.min(), series.max()

        return inlier_series.min(), inlier_series.max()

    @staticmethod
    def get_gap_differences(data):

        differences = []
        for i, value in enumerate(data):

            if i < len(data) - 1:
                difference = value - data[i + 1]
                if difference < 0:
                    difference = difference * -1
                differences.append(difference)

        return differences

    @staticmethod
    def check_ocr(lines_with_styling: list) -> bool:

        if len(lines_with_styling) == 0:
            return False

        words = 1
        phrases = 1

        for i, line in enumerate(lines_with_styling):

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

    @staticmethod
    def get_word_gaps(lines, threshold):
        i = 0
        origin_x_differences = []
        while i < len(lines):
            current_word = lines[i]
            line_y_boundary = round(current_word.origin_y)
            current_line = []

            while i < len(lines) and round(lines[i].origin_y) == line_y_boundary:
                current_line.append(lines[i])
                i += 1
            origin_x_differences.extend(DocumentAnalysis.get_gap_differences([round(line.origin_x) for line in current_line]))
        lower_bound, upper_bound = DocumentAnalysis.get_styling_bounds(sorted(origin_x_differences), threshold=threshold)

        return lower_bound, upper_bound

    @staticmethod
    def get_line_gaps(style_counter, threshold):
        values = sorted(style_counter, reverse=True)
        differences = DocumentAnalysis.get_gap_differences(values)
        lower_bound, upper_bound = DocumentAnalysis.get_styling_bounds(differences, threshold=threshold)

        return lower_bound, upper_bound

    @staticmethod
    def get_min_styling_attr(style_counter):
        return min(style_counter)

    @staticmethod
    def get_max_styling_attr(style_counter):
        return max(style_counter)

    @staticmethod
    def get_page_heuristics(lines: list, ocr: bool) -> dict:

        font_size_counter = DocumentAnalysis.get_styling_counter(lines_with_styling=lines, styling_attribute="font_size")
        font_name_counter = DocumentAnalysis.get_styling_counter(lines_with_styling=lines, styling_attribute="font_name")
        origin_x_counter = DocumentAnalysis.get_styling_counter(lines_with_styling=lines, styling_attribute="origin_x")
        origin_y_counter = DocumentAnalysis.get_styling_counter(lines_with_styling=lines, styling_attribute="origin_y")

        most_common_font_size = DocumentAnalysis.get_n_most_common(counter=font_size_counter, n=1)
        most_common_font_name = DocumentAnalysis.get_n_most_common(counter=font_name_counter, n=1)
        most_common_origin_x = DocumentAnalysis.get_n_most_common(counter=origin_x_counter, n=1)

        if ocr:
            threshold = 3.0
        else:
            threshold = 1.0

        font_size_expanded_data = []
        for value, freq in sorted(font_size_counter.items()):
            font_size_expanded_data.extend([value] * freq)

        font_size_lower_bound, font_size_upper_bound = DocumentAnalysis.get_styling_bounds(font_size_expanded_data, threshold=threshold)


        origin_x_lower_bound, origin_x_upper_bound = DocumentAnalysis.get_word_gaps(lines=lines, threshold=threshold)
        origin_y_lower_bound, origin_y_upper_bound = DocumentAnalysis.get_line_gaps(style_counter=origin_y_counter, threshold=threshold)


        return {'origin x': {'most common': most_common_origin_x[0][0], 'lower bound': origin_x_lower_bound, 'upper bound': origin_x_upper_bound},
                'origin y': {'maximum': DocumentAnalysis.get_max_styling_attr(style_counter=origin_y_counter), 'lower bound': origin_y_lower_bound, 'upper bound': origin_y_upper_bound},
                'font size': {'most common': most_common_font_size[0][0], 'lower bound': font_size_lower_bound, 'upper bound': font_size_upper_bound},
                'font name': {'most common': most_common_font_name[0][0]}}

    @staticmethod
    def filter_by_boundaries(lines_with_styling, ocr: bool):

        filtered_lines: list[StyledLine] = []
        current_line = []

        font_heuristics = DocumentAnalysis.get_page_heuristics(lines=lines_with_styling, ocr=ocr)

        left_boundary = font_heuristics['origin x']['most common']
        top_boundary = None
        bottom_boundary = font_heuristics['origin y']['maximum']

        i = 0
        while i < len(lines_with_styling):

            current_word = lines_with_styling[i]
            line_y_boundary = round(current_word.origin_y)

            if top_boundary is None: # Removing headers
                
                if round(current_word.origin_x) < left_boundary: # Header
                    while round(lines_with_styling[i].origin_y) == line_y_boundary:
                        i += 1

                elif round(current_word.origin_x) == left_boundary:  # Body start

                    gap_to_next_line = 0
                    j = i
                    while gap_to_next_line == 0:
                        gap_to_next_line = round(lines_with_styling[j + 1].origin_y) - round(lines_with_styling[i].origin_y)
                        j += 1

                    if gap_to_next_line > font_heuristics['origin y']['upper bound']: # Aligned header
                        while round(lines_with_styling[i].origin_y) == line_y_boundary:
                            i += 1
                    else:
                        top_boundary = round(current_word.origin_y)
                        while round(lines_with_styling[i].origin_y) == line_y_boundary:
                            current_line.append(lines_with_styling[i])
                            i += 1

                elif round(current_word.origin_x) > left_boundary: # Edge case: Indented main body

                    if round(current_word.origin_x) - font_heuristics['origin x']['lower bound'] > left_boundary: # Indented header
                        while round(lines_with_styling[i].origin_y) == line_y_boundary:
                            i += 1

                    elif font_heuristics['font size']['lower bound'] <= round(current_word.font_size) <= font_heuristics['font size']['upper bound']:

                        top_boundary = round(current_word.origin_y)

                        while round(lines_with_styling[i].origin_y) == line_y_boundary:
                            current_line.append(lines_with_styling[i])
                            i += 1

                    else:
                        while round(lines_with_styling[i].origin_y) == line_y_boundary:
                            i += 1

                else:
                    while round(lines_with_styling[i].origin_y) == line_y_boundary:
                        i += 1

            elif lines_with_styling[i].origin_y >= bottom_boundary - font_heuristics['origin y']['lower bound']: # Very bottom

                if round(lines_with_styling[i].origin_x) == round(filtered_lines[-1].origin_x): # Continued indented block

                    while round(lines_with_styling[i].origin_y) == line_y_boundary:

                        if i == len(lines_with_styling) - 1:
                            current_line.append(lines_with_styling[i])
                            i += 1
                            break
                        else:
                            word_separation = round(lines_with_styling[i + 1].origin_x) - round(lines_with_styling[i].origin_x)

                            if font_heuristics['origin x']['lower bound'] <= word_separation <= font_heuristics['origin x']['upper bound']: # Doesn't work for non-ocr
                                current_line.append(lines_with_styling[i])
                                i += 1
                            else: # Replace with table detection
                                current_line.append(lines_with_styling[i])
                                i += 1
                                break

                else:
                    while i < len(lines_with_styling) and round(lines_with_styling[i].origin_y) == line_y_boundary: # Footer
                        i += 1

            elif round(current_word.origin_x) == left_boundary: # Main body

                if font_heuristics['font size']['lower bound'] <= round(lines_with_styling[i].font_size) <= font_heuristics['font size']['upper bound']:
                    while round(lines_with_styling[i].origin_y) == line_y_boundary:
                        current_line.append(lines_with_styling[i])
                        i += 1

                else: # Edge case: Aligned title
                    while round(lines_with_styling[i].origin_y) == line_y_boundary:
                        i += 1

            elif round(lines_with_styling[i].origin_y) < round(filtered_lines[-1].origin_y): # Titles outside regular read-order

                if lines_with_styling[i].font_size == font_heuristics['font size']['most common']: # Edge case: Indented main body
                    while round(lines_with_styling[i].origin_y) == line_y_boundary:
                        current_line.append(lines_with_styling[i])
                        i += 1
                else:
                    while i < len(lines_with_styling) and round(lines_with_styling[i].origin_y) == line_y_boundary:
                        i += 1

            elif round(current_word.origin_x) > left_boundary: # Indented block

                if i == len(lines_with_styling) - 1 and round(lines_with_styling[i].origin_y) == line_y_boundary:
                    current_line.append(lines_with_styling[i])
                    i += 1

                elif i < len(lines_with_styling) - 1:
                    while round(lines_with_styling[i].origin_y) == line_y_boundary:

                        word_separation = round(lines_with_styling[i + 1].origin_x) - round(lines_with_styling[i].origin_x)

                        if font_heuristics['origin x']['lower bound'] <= word_separation <= font_heuristics['origin x']['upper bound']:
                            current_line.append(lines_with_styling[i])
                            i += 1
                        else:
                            current_line.append(lines_with_styling[i])
                            i += 1
                            break

            elif round(current_word.origin_x) < left_boundary:  # Left-side footer
                while i < len(lines_with_styling) and round(lines_with_styling[i].origin_y) == line_y_boundary:
                    i += 1
            else:
                i += 1


            if len(current_line) > 0:
                # for word in current_line:

                filtered_lines.append(StyledLine(text=' '.join(line.text for line in current_line if line.text.strip()),
                                                 font_size=round(pd.Series([line.font_size for line in current_line]).mean()),
                                                 font_name=current_word.font_name,
                                                 origin_x=round(current_word.origin_x),
                                                 origin_y=round(current_word.origin_y)))
                current_line = []


        return filtered_lines

    # @staticmethod
    # def separate_by_paragraph(lines_with_styling: list):

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
        for page in range(pdf_reader.get_page_count()):

            page_blocks = DocumentAnalysis.get_page_blocks_from_dict(pdf=pdf_reader.pdf, page_number=page, sort=False)
            lines_with_styling = DocumentAnalysis.get_pdf_styling_from_blocks(page_blocks=page_blocks)
            lines_without_blanks = [line for line in lines_with_styling if line.text.strip()]

            ocr = False
            if DocumentAnalysis.check_ocr(lines_with_styling=lines_with_styling):
                ocr = True

            lines_with_styling = DocumentAnalysis.filter_by_boundaries(lines_with_styling=lines_without_blanks, ocr=ocr)
            lines_without_numbers = Cleaner.clean_page_numbers(lines=lines_with_styling)
            cleaned_text = Cleaner.join_broken_sentences(lines=lines_without_numbers)
            page_text, multipage_parentheses = Cleaner.clean_extracted_text(text=cleaned_text,
                                                                                 multipage_parentheses=multipage_parentheses, ocr=ocr)

            output_writer.write(mode="a", text=f'{page_text}\n\n')

    pdf_reader.close()

if __name__ == '__main__':
    main()