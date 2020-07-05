#!pip install PyPDF2

import os
from PyPDF2 import PdfFileReader, PdfFileWriter


def pdf_splitter(path, output_path):
    fname = os.path.splitext(os.path.basename(path))[0]
    fname = output_path + fname
    pdf = PdfFileReader(path)
    for page in range(pdf.getNumPages()):
        pdf_writer = PdfFileWriter()
        pdf_writer.addPage(pdf.getPage(page))
        output_filename = '{}_{}.pdf'.format(
            fname, page)
        with open(output_filename, 'wb') as out:
            pdf_writer.write(out)
        print('Created: {}'.format(output_filename))


if __name__ == '__main__':
    path = '/home/jupyter/nafize/data'
    output_path = '/home/jupyter/nafize/data/splited_pdf'
    for filename in os.listdir(path):
        if filename.endswith(".pdf"):
            pdf_splitter(os.path.join(path, filename), output_path)