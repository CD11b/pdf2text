import sys
import pymupdf
import os
import re
import unicodedata
from dataclasses import dataclass
from collections import Counter
import logging

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

    # @staticmethod
    # def generate_ocr_lines(lines_with_styling: list, most_common_font_name: str) -> list:
    #
    #     logging.debug(f"Joining broken sentences of OCR'd text.")
    #
    #     try:
    #
    #         joined_lines = []
    #         line = 0
    #
    #         for i, current_line in enumerate(lines_with_styling):
    #
    #             if i == 0:
    #
    #                 joined_lines.append(current_line.text)
    #
    #             else:
    #                 if current_line.origin_y == joined_lines[-1].origin_y:
    #                     joined_lines.append(current_line.text)
    #                 else:
    #                     line += 1
    #                     joined_lines.append(current_line.text)
    #
    #     except Exception as e:
    #         logging.exception(f"Error joining broken sentences: {e}")
    #         raise


    @staticmethod
    def clean_extracted_text(text: str, multipage_parentheses: str) -> tuple[str, str]:
        """Clean text in a single pass for better performance."""

        try:
            result = []
            i = 0

            while i < len(text):
                char = text[i]

                # Skip parentheses and their content
                pairs = {'(': ')', '[': ']', '{': '}'}

                if char in pairs or multipage_parentheses:

                    open_char = char
                    if multipage_parentheses:
                        open_char = multipage_parentheses

                    close_char = pairs[open_char]
                    depth = 1
                    i += 1
                    while i < len(text) and depth > 0:
                        if text[i] == open_char:
                            depth += 1
                        elif text[i] == close_char:
                            depth -= 1
                            multipage_parentheses = None
                        elif i == len(text) - 1:
                            multipage_parentheses = open_char

                        i += 1

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
    def get_page_blocks_from_dict(pdf: pymupdf.Document, page_number: int) -> list:

        try:
            page_dict = pdf[page_number].get_text("dict") #read page 1
            page_blocks = page_dict["blocks"]

            return page_blocks

        except Exception as e:
            logging.exception(f"Error reading PDF blocks: {e}")
            raise

    @staticmethod
    def get_pdf_styling_from_blocks(page_blocks: list) -> list:

        try:
            lines_with_styling: list[StyledLine] = []

            for block in sorted(page_blocks, key=lambda b: (b['bbox'][1], b['bbox'][0])):  # top-left to bottom-right
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
    def get_styling_frequency(lines_with_styling: list, styling_attribute: str) -> Counter:

        try:
            counter = Counter()
            for line in lines_with_styling:
                attr_value = getattr(line, styling_attribute)
                counter[attr_value] += len(line.text)
            return counter

        except Exception as e:
            logging.exception(f"Error calculating style frequencies for {styling_attribute}: {e}")
            raise

    @staticmethod
    def get_n_most_common(counter: Counter, n: int) -> list:

        try:
            return counter.most_common(n)
        except Exception as e:
            logging.exception(f"Error finding most common style: {e}")
            raise

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
    def get_font_heuristics(lines_with_styling: list) -> dict:

        font_size_frequency = DocumentAnalysis.get_styling_frequency(lines_with_styling=lines_with_styling, styling_attribute="font_size")
        font_name_frequency = DocumentAnalysis.get_styling_frequency(lines_with_styling=lines_with_styling, styling_attribute="font_name")

        most_common_font_size = DocumentAnalysis.get_n_most_common(counter=font_size_frequency, n=1)
        most_common_font_name = DocumentAnalysis.get_n_most_common(counter=font_name_frequency, n=1)

        return {'font size': {'frequencies': font_size_frequency, 'most common': most_common_font_size[0][0]}, 'font name': {'frequencies': font_name_frequency, 'most common': most_common_font_name[0][0]}}

    @staticmethod
    def filter_dominant_font(lines_with_styling: list) -> list:

        if len(lines_with_styling) == 0:
            return lines_with_styling

        font_heuristics = DocumentAnalysis.get_font_heuristics(lines_with_styling)

        # if DocumentAnalysis.check_ocr(lines_with_styling=lines_with_styling):
        #     x = True
        #     Cleaner.generate_ocr_lines(lines_with_styling=lines_with_styling, most_common_font_name=font_heuristics['font name']['most common'])

        filtered_lines = []

        for i, line in enumerate(lines_with_styling):

            if i == 0:
                pass

            if line.font_size != font_heuristics['font size']['most common']:
                continue
            elif line.font_name != font_heuristics['font name']['most common']:

                if i > 0 and len(filtered_lines) >= 1:
                    last_line = filtered_lines[-1]
                    if line.origin_y == last_line.origin_y and last_line.font_name == font_heuristics['font name']['most common']:
                        filtered_lines.append(line)
                        continue

                if i < len(lines_with_styling) - 1:
                    next_line = lines_with_styling[i + 1]
                    if line.origin_y == next_line.origin_y and next_line.font_name == font_heuristics['font name']['most common']:
                        filtered_lines.append(line)
            else:

                if i == 0: # Header
                    next_line = lines_with_styling[i + 1]
                    if line.origin_y != next_line.origin_y:
                        if line.origin_x == next_line.origin_x:
                            pass
                        else:
                            continue

                if i == len(lines_with_styling) - 1: # Footer
                    last_line = filtered_lines[-1]
                    if line.origin_y != last_line.origin_y:
                        continue

                filtered_lines.append(line)

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
            self.output_path = f"{sanitized_title}.txt"

        else:
            base_name = os.path.splitext(os.path.basename(pdf_path))[0]
            self.output_path = f"{base_name}.txt"

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

        multipage_parentheses = False
        for page in range(pdf_reader.get_page_count()):

            page_blocks = DocumentAnalysis.get_page_blocks_from_dict(pdf=pdf_reader.pdf, page_number=page)
            lines_with_styling = DocumentAnalysis.get_pdf_styling_from_blocks(page_blocks=page_blocks)
            lines_without_numbers = Cleaner.clean_page_numbers(lines=lines_with_styling)
            filtered_lines_with_styling = DocumentAnalysis.filter_dominant_font(lines_with_styling=lines_without_numbers)
            cleaned_text = Cleaner.join_broken_sentences(lines=filtered_lines_with_styling)
            page_text, multipage_parentheses = Cleaner.clean_extracted_text(text=cleaned_text,
                                                                                 multipage_parentheses=multipage_parentheses)

            output_writer.write(mode="a", text=f'{page_text}\n\n')

    pdf_reader.close()

if __name__ == '__main__':
    main()