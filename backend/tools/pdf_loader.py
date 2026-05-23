import os


def load_pdf(file_path):

    if not os.path.exists(file_path):
        raise Exception("PDF file not found")

    return file_path