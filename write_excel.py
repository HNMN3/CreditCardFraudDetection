import xlsxwriter
import os
from pandas import Series


class DataExplorationSheet(object):
    def __init__(self, data_row, filename='data_exploration.xlsx', header=None):
        assert type(data_row) == Series

        if header is None:
            header = ['Variable Name', 'Type', 'Segment', 'Expectation', 'Conclusion', 'Comments']
        assert type(header) == list

        self.header = header
        self.row = data_row
        self.filename = filename

    def save(self):
        if os.path.exists(self.filename):
            raise Exception('File already exist!!')

        data = []
        for index in self.row.index:
            data.append((index, type(self.row[index])))

        workbook = xlsxwriter.Workbook(self.filename)
        worksheet = workbook.add_worksheet()

        i = 0
        j = 0
        # put header
        for heading in self.header:
            worksheet.write(i, j, heading)
            j += 1

        i += 1

        for name, vtype in data:
            worksheet.write(i, 0, str(name))
            worksheet.write(i, 1, str(vtype))
            i += 1

        workbook.close()
