from typing import List, Union, Optional, TypeVar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pytesseract
import os
import re
import cv2
from itertools import zip_longest

if os.name == "nt":
    pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'


class ExtractTable():
    """
    Module to to extract table and contents from a image.

    Attributes:

    tesseract_config: 'dict' Tesseract Config used to extract data.
    expected_columns: 'int' Expected Column count for each Table containing marks.
    self.row_lines_config: 'dict[subImagename] = List[Tuple(start_line_index, end_line,index)]'
    """
    def __init__(self, debug=True) -> None:
        """
        debug: Set to True to see the image processing filterations applied 
                to extract required information.
        """
        self.debug = debug
        if debug == True:
            self.imshow = cv2.imshow
        else:
            self.imshow = lambda x: None

        self.tesseract_config = {
            "config": '--psm 12 --oem 1',
            "output_type": "data.frame",
            "lang": "eng"
        }

        self.expected_columns = 5

        self.row_lines_config = {
            "HeadingColumn": [(0, 1)],
            "Semesters": [(1, 2), (5, 6), (9, 10)],
            "Institutes": [(2, 3), (6, 7), (10, 11)],
            "Grades": [(3, 4), (7, 8), (11, 12)],
            "GPAs": [(4, 5), (8, 9)]
        }

    def extract_data(self, img: np.ndarray) -> pd.DataFrame:
        """
        Method to extract table data from a image using pytesseract.
        """
        data = pytesseract.image_to_data(img, **self.tesseract_config)
        return data

    def extract_string(self, img: np.ndarray) -> str:
        """
        Method to extract string data from a image using pytesseract. 
        """
        data = pytesseract.image_to_string(img)  # , **self.tesseract_config)
        return data

    def parse_table_lines(self, img: np.ndarray) -> List:
        """
        Extract Horizontal lines using Hugh transform and applying filter
        to extract boundaries of each table, sections, and heading.
        """
        # Convert the img to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply edge detection method on the image
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # This returns an array of r and theta values
        lines = cv2.HoughLines(edges, 1, np.pi/180, 200)

        filtered_lines = []

        # The below for loop runs till r and theta values
        # are in the range of the 2d array
        for r_theta in lines:
            arr = np.array(r_theta[0], dtype=np.float64)
            r, theta = arr
            # Stores the value of cos(theta) in a
            # print(r, theta, type(theta))
            a = np.cos(theta)

            # Stores the value of sin(theta) in b
            b = np.sin(theta)

            # x0 stores the value rcos(theta)
            x0 = a*r

            # y0 stores the value rsin(theta)
            y0 = b*r

            # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
            x1 = int(x0 + 1000*(-b))

            # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
            y1 = int(y0 + 1000*(a))

            # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
            x2 = int(x0 - 1000*(-b))

            # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
            y2 = int(y0 - 1000*(a))

            # Compute Line angle
            angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180. / np.pi)

            if angle > 0.1:
                # exclude vertical lines
                continue
            else:
                filtered_lines.append((x1, y1, x2, y2))

        filtered_lines = self._filter_closer_lines(filtered_lines)
        return filtered_lines

    def _filter_closer_lines(self, lines: List, line_distance_threshold: int = 5) -> List:
        """
        Method to filter lines close to each other based on  threshold.
        """
        # Sort by y1
        lines.sort(key=lambda x: x[1])
        # Pre Filter lines close to Top and bottom # Thresholds found emperically
        lines = lines[3:-2]

        filtered_lines = []
        filtered_lines.append(lines[0])
        prev = (lines[0][1], lines[0][3])
        for line in lines[1:]:
            # if y1 or y2 distance to previous detected y1 and y2 is smaller thean threshold exclude the line
            if abs(line[1] - prev[0]) < line_distance_threshold or abs(line[3] - prev[1]) < line_distance_threshold:
                continue
            else:
                filtered_lines.append(line)
                prev = (line[1], line[3])
        return filtered_lines

    def plot_lines(self, img, lines):
        """
        Plot detected lines seperating regions of interest in debug mode

        Raises RuntimeError: Incase called without debug mode.
        """
        if not self.debug:
            raise RuntimeError("Object not initiated in debug mode!")

        img = img.copy()
        for line in lines:
            x1, y1, x2, y2 = line
            # Draw  red line.
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        self.imshow(img)

    def extract_sub_image(self, img: np.ndarray, filtered_lines: List, image_section: str) -> np.ndarray:
        """
        Method to extract a subimage and yield based on inputted image_section and filtered_lines list.

        Raise ValueError: If provided image_section arg input is not in row_lines_config declared.
        """
        if image_section not in self.row_lines_config.keys():
            raise ValueError("Invalid image_section Key!")
        
        for indexs in self.row_lines_config.get(image_section):
            line1 = filtered_lines[indexs[0]]
            line2 = filtered_lines[indexs[1]]
            x_s = min(line1[1], line2[1])
            x_e = max(line1[3], line2[3])
            y_s = -1000
            y_e = 999
            im = img[x_s:x_e, y_s:y_e].copy()
            yield im

    def extract_column_names(self, img: np.ndarray, filtered_lines: List) -> List:
        """
        Method to specifically extract column names from Heading Section of image.
        """
        # Extract sub image with column names
        im = next(self.extract_sub_image(img, filtered_lines, "HeadingColumn"))
        self.imshow(im)

        # Extract table from Image
        data = self.extract_data(im)

        # Filter by confidence
        data = data[data["conf"] > 20]

        column_names = []
        for left, text in zip(data['left'], data['text']):
            if re.search(r'[a-zA-Z]+', text):
                if "/" in text:
                    # Wild Card to seperate the column by "/"
                    sep = text.split("/")
                    column_names.append(sep[0])
                    column_names.append(sep[1])
                else:
                    column_names.append(text)

        # Wild Card to Handle Course  Title
        try:
            ind1 = column_names.index("Course")
            ind2 = column_names.index("Title")
            if ind2 == ind1 + 1:
                del column_names[ind2]
                column_names[ind1] = "Course-Title"
        except ValueError:
            pass

        return column_names

    def clear_images(self, image:np.ndarray) -> np.ndarray:
        """
        Method to clear the image with artifacts on boundary 
        and apply otsu binarization for better detection when fed to
        tesseract module
        """
        self.imshow(image)
        # Clip  Left and right boundaries by 4%
        bound = int(0.04*image.shape[1])
        image[:, :bound] = 255
        image[:, (image.shape[1] - bound):] = 255
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply Otsu binariztion
        (threshi, img_bw) = cv2.threshold(img_gray,
                                          0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        img_bw[img_bw < 200] = 0
        img_bw[img_bw >= 200] = 255
        self.imshow(img_bw)
        return img_bw

    def extract_semesters(self, image: np.ndarray, filtered_lines: List) -> List[str]:
        """
        Method specifically designed to extract semester names
        """
        semesters = []
        for im in self.extract_sub_image(image, filtered_lines, "Semesters"):
            im = self.clear_images(im)
            data = self.extract_string(im)
            data = " ".join(data.split())
            semesters.append(data)
        return semesters

    def extract_gpas(self, image: np.ndarray, filtered_lines: List) -> List[dict]:
        """
        Method specifically designed to extract GPAs per semester
        """
        gpas = []
        for im in self.extract_sub_image(image, filtered_lines, "GPAs"):
            im = self.clear_images(im)
            data = self.extract_string(im).strip()
            data = data.split("0 ")
            scores = {}
            for i in data:
                score = i.replace(" ", "").split(":")
                scores[score[0]] = float(score[1])
            gpas.append(scores)
        return gpas

    def extract_marks(self, image: np.ndarray, filtered_lines: List, column_names: List[str], exclusion_column_names: List[str] = ["Mark"]) -> List[pd.DataFrame]:
        """
        Method specifically designed to extract Marks per semester and return list of pandas dfs per semester.

        Parameters:
        -----------
        image: np.ndarray input image in BGR
        column_names: List[str] Column Names extracted from the whole image.
        exclusion_column_names: List[str] Column Names we are failing to extract completely.

        Returns:
        --------
            marks: List[pd.DataFrame] Per semester Marks Data Frame.
        """
        # Wild card to remove unextractable Mark
        for exl in exclusion_column_names:
            column_names.remove(exl)

        marks = []
        for im in self.extract_sub_image(image, filtered_lines, "Grades"):
            im = self.clear_images(im)
            data = self.extract_data(im)
            data = data[data["conf"] > 10]
            data = data.sort_values(by=['top', 'left'], ascending=[True, True])
            data.reset_index()
            temp = data['left'].shift(-1).copy()
            data['left+width'] = data['left'] + data['width']
            data["distance_to_next_word"] = temp - data['left+width']
            data = data.reset_index()

            # create a new column to hold the merged values
            data['merged'] = data['text']

            # loop through the dataframe and merge rows
            for i in range(len(data)-1):
                if data.loc[i, 'distance_to_next_word'] < 10 and data.loc[i, 'distance_to_next_word'] > 0:
                    data.loc[i, 'merged'] += ' ' + data.loc[i+1, 'text']
                    data.loc[i+1, 'merged'] = ''

            # drop rows with empty merged values
            data = data[data['merged'] != '']
            data = data[data['merged'] != '+p']

            # drop the original text and distance columns
            data = data[['merged']]
            data = data[data['merged'] != '|']

            def exclusions(x): return re.search(
                r'[0-9]+', x) or re.search(r'[a-zA-Z]+', x)

            elements = [i for i in data["merged"] if exclusions(i)]

            if len(elements) % len(column_names) > 0:
                elements = elements[:-(len(elements) % len(column_names))]

            arr = np.array(elements).reshape(-1, len(column_names))
            df = pd.DataFrame(arr, columns=column_names)
            marks.append(df)
        return marks

    def extract_table(self, img_path: str) -> dict:
        """
        Method serves as serving function for module. Return information
        dict containting all extracted information as dict with keys as semester names
        and value and GPA info and Marks pandas DataFrame

        Parameters:
        -----------
            img_path: (str) Path to image

        Returns:
        --------
            table_info: dict key: 'str' Semester Name
                             Val: 'dict' Key: 'GPA_Info'
                                         Val: 'dict'
                                         
                                         key: 'Marks'
                                         Val: 'pd.DataFrame' Pandas Dataframe.

        Exceptions:
        -----------
            ValueError: Raised if input image file doesn't exist.
        """
        if not os.path.isfile(img_path):
            raise ValueError(f"Input image file doesn't exist: {img_path}")

        img_bgr = cv2.imread(img_path)
        if not isinstance(img_bgr,np.ndarray):
            raise RuntimeError("Corrupted Image!")
        
        filtered_lines = self.parse_table_lines(img_bgr)
        if self.debug:
            self.plot_lines(img_bgr.copy(), filtered_lines)
        semester_info = self.extract_semesters(img_bgr, filtered_lines)
        gpas_info = self.extract_gpas(img_bgr, filtered_lines)
        column_names = self.extract_column_names(img_bgr, filtered_lines)
        marks_df = self.extract_marks(
            img_bgr, filtered_lines, column_names, ["Mark"])

        table_info = {}

        for sem, gpa, df_ in zip_longest(semester_info, gpas_info, marks_df):
            table_info[sem] = {
                "GPA_Info": gpa,
                "Marks": df_}
        return table_info


if __name__ == "__main__":
    import argparse
    import pprint
    parser = argparse.ArgumentParser(
        "Module to take a image as input in constrainted format and extract table contents from it.  \
            See \TableDataExtraction\flexdaytestdata.png for sample image.")

    # add the command-line arguments
    parser.add_argument("image_file", nargs="?", default="./flexdaytestdata.png",
                    help="path to the image file (default: %(default)s)")
    parser.add_argument('--debug', action='store_true',
                        help='enable debug mode')
    parser.add_argument('--output_file', type=str,
                        help='path to output file (default: %(default)s)', default = "./output.npy")

    # parse the arguments
    args = parser.parse_args()

    # access the parsed arguments
    output_file = args.output_file
    engine_obj = ExtractTable(debug=args.debug)
    ret = engine_obj.extract_table(args.image_file)
    pprint.pprint(ret)
    np.save(output_file, ret)
    pass
